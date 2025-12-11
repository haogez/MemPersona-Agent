from __future__ import annotations

import textwrap
from typing import Dict, Any


def build_persona_prompt(seed: str) -> list[dict[str, str]]:
    system = textwrap.dedent(
        """
        你是一个擅长角色塑造的作家，根据用户的一句话，生成细粒度 20 维角色人设。要求：
        - 结果必须是严格的 JSON，对应字段：name, age, gender, occupation, hobby, skill, values, living_habit, dislike, language_style, appearance, family_status, education, social_pattern, favorite_thing, usual_place, past_experience, background, speech_style, personality{openness, conscientiousness, extraversion, agreeableness, neuroticism}
        - 所有文本使用中文，数字使用阿拉伯数字。
        - 不得添加解释文字，只输出 JSON。
        """
    ).strip()
    user = f"用户设定：{seed}。请输出 JSON。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_episode_prompt(persona: Dict[str, Any]) -> list[dict[str, str]]:
    persona_desc = "\n".join([f"{k}: {v}" for k, v in persona.items() if k != "personality"])
    personality = persona.get("personality", {})
    personality_desc = ", ".join([f"{k}={v}" for k, v in personality.items()])
    system = textwrap.dedent(
        f"""
        你是一名传记作家，根据以下角色人设生成 3-5 条静态人生记忆 Episode。要求：
        - 年龄升序排列
        - narrative 采用第一人称叙述，200-300 字
        - 所有字段使用中文
        - 输出严格的 JSON，结构见 Episode 规范
        角色人设：
        {persona_desc}
        性格分布：{personality_desc}
        """
    ).strip()
    return [{"role": "system", "content": system}]
