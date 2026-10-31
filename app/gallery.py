"""会话内的多模板身份库；仅保存特征和小缩略图内存，不把用户内容写入磁盘。"""
from dataclasses import dataclass, field

import numpy as np

from .reid import cosine, normalize
from .schemas import IdentitySummary


@dataclass
class Identity:
    """一个登记身份，模板数量有限；公开摘要不包含模板或原图。"""
    id: str
    name: str
    source: str
    templates: list = field(default_factory=list)
    thumbnail: bytes | None = None
    distractors: list = field(default_factory=list)


class Gallery:
    """输入注册特征/待匹配特征，输出身份编号或最优与次优相似度。"""

    def __init__(self, config):
        self.config = config
        self.items = {}
        self.next_id = 1

    def register(self, feature, name="长按锁定", source="lock", thumbnail=None, distractors=()):
        """登记一个特征，返回新身份；上限检查由会话转成业务错误。"""
        if len(self.items) >= self.config.identity_limit:
            raise ValueError("身份数量已达上限，请先删除不用的身份")
        identity_id = f"ID-{self.next_id:02d}"
        self.next_id += 1
        identity = Identity(identity_id, name, source, [normalize(feature).copy()], thumbnail)
        # 反例来自用户登记时可见的其他人，只留有限特征，既不保存裁剪也不返回给页面。
        identity.distractors = [normalize(item).copy() for item in list(distractors)[:self.config.reid_max_templates]]
        self.items[identity_id] = identity
        return identity

    def score(self, feature, identity_id):
        """将输入特征与该身份的多个模板逐一比较，返回最高相似度。"""
        return max(cosine(feature, template) for template in self.items[identity_id].templates)

    def supports(self, feature, identity_id):
        """输入本帧特征与身份号，输出布尔结果；判断规则由assess统一实现。"""
        return self.assess(feature, identity_id)["accepted"]

    def assess(self, feature, identity_id):
        """输入特征和身份号，输出通过与否、原因及少量分数，供会话诊断使用。

        计算仍使用原始精度；只在输出诊断时取四位小数，不输出向量或图片，也不放宽任何门槛。
        """
        identity = self.items[identity_id]
        anchor = identity.templates[0]
        score, anchor_score = self.score(feature, identity_id), cosine(feature, anchor)
        values = {"score": score, "anchor_score": anchor_score, "high": self.config.reid_tau_high,
                  "color_min": self.config.reid_color_min, "margin": self.config.reid_margin}
        reason = "ACCEPTED"
        if score < self.config.reid_tau_high:
            reason = "FUSED_SCORE_LOW"
        elif anchor_score < self.config.reid_tau_high:
            reason = "ANCHOR_SCORE_LOW"
        if len(feature) in {704, 640} and len(feature) == len(anchor):
            split = len(feature) - 128
            # 权重为0时对应分量是零向量，不计算其余弦，避免除以0。
            if self.config.reid_w_appear:
                values["appearance_score"] = cosine(feature[:split], anchor[:split])
                if reason == "ACCEPTED" and values["appearance_score"] < self.config.reid_tau_high:
                    reason = "APPEARANCE_SCORE_LOW"
            if self.config.reid_w_color:
                values["color_score"] = cosine(feature[split:], anchor[split:])
                if reason == "ACCEPTED" and values["color_score"] < self.config.reid_color_min:
                    reason = "COLOR_SCORE_LOW"
        if identity.distractors:
            nearest_other = max(cosine(feature, item) for item in identity.distractors)
            values["nearest_other"] = nearest_other
            values["other_margin"] = anchor_score - nearest_other
            if reason == "ACCEPTED" and values["other_margin"] < self.config.reid_margin:
                reason = "DISTRACTOR_TOO_CLOSE"
        return {"accepted": reason == "ACCEPTED", "reason": reason,
                **{key: round(value, 4) for key, value in values.items()}}

    def rank(self, feature, excluded=()):
        """保留完整身份库的第一/第二名来做 margin，在线身份也参与歧义比较。

        excluded 身份不能再绑定给其他轨迹，但不能从第二名竞争中消失，否则可能误认次优身份。
        """
        scores = sorted(((self.score(feature, key), key) for key in self.items), reverse=True)
        if not scores or scores[0][1] in excluded:
            return None, -1.0, -1.0
        return scores[0][1], scores[0][0], scores[1][0] if len(scores) > 1 else -1.0

    def append(self, identity_id, feature):
        """输入确认匹配的特征，保留原始注册模板，再用有限滚动模板补充新姿态。

        返回是否实际增加模板，供离线诊断区分“满足更新条件”和“模板确实改变”。
        """
        identity = self.items[identity_id]
        if self.config.reid_max_templates <= 1:
            return False
        if max(cosine(feature, template) for template in identity.templates) > 0.995:
            return False
        identity.templates.append(normalize(feature).copy())
        if len(identity.templates) > self.config.reid_max_templates:
            del identity.templates[1]
        return True

    def update_ema(self, identity_id, feature, alpha, appearance_dimensions=None):
        """用 EMA 更新单个动态槽，保留第一个锚点并可只平滑 OSNet 身体分量。

        输入已确认的特征、当前观测权重 alpha 和可选的身体特征维数；输出是否更新、
        跳过原因及模板数。B1/B2 每个身份只用 templates[0] 锚点与 templates[1] 动态槽，
        两组的匹配仍由 score() 对这些模板取最高余弦相似度。
        """
        identity = self.items[identity_id]
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not np.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("EMA alpha 必须是 0 到 1 之间的有限数")
        if self.config.reid_max_templates <= 1:
            return {"updated": False, "reason": "template_capacity_one",
                    "template_count": len(identity.templates)}
        incoming = normalize(feature)
        anchor = identity.templates[0]
        if incoming.shape != anchor.shape:
            raise ValueError("EMA 输入特征维度必须与初始模板一致")
        if appearance_dimensions is None:
            appearance_dimensions = len(incoming)
        if (type(appearance_dimensions) is not int or not 1 <= appearance_dimensions <= len(incoming)):
            raise ValueError("appearance_dimensions 必须在特征向量范围内")
        if alpha == 0:
            return {"updated": False, "reason": "zero_quality_weight",
                    "template_count": len(identity.templates)}

        # 第一份动态模板从原始锚点开始，确保 B1/B2 使用完全相同的更新结构。
        old = identity.templates[1] if len(identity.templates) > 1 else anchor
        old_body = normalize(old[:appearance_dimensions])
        new_body = normalize(incoming[:appearance_dimensions])
        body_scale = float(np.linalg.norm(old[:appearance_dimensions]))
        mixed_body = normalize((1.0 - float(alpha)) * old_body + float(alpha) * new_body) * body_scale
        if appearance_dimensions < len(incoming):
            # 只更新 OSNet 部分；拼接向量后面的颜色分量保持锚点值，避免把颜色变化混称 OSNet 质量。
            updated = np.concatenate([mixed_body, old[appearance_dimensions:]])
        else:
            updated = mixed_body
        updated = normalize(updated)
        if len(identity.templates) > 1:
            identity.templates[1] = updated
        else:
            identity.templates.append(updated)
        return {"updated": True, "reason": "ema_applied", "template_count": len(identity.templates)}

    def summaries(self, online):
        """输入本帧在线身份集合，输出页面摘要，不泄露内部向量。"""
        return [IdentitySummary(id=item.id, name=item.name, online=item.id in online)
                for item in self.items.values()]
