from __future__ import annotations

import textwrap
import json
from typing import Any, Dict, List


def build_persona_prompt(description: str, timeline_mode: str = "strict") -> List[Dict[str, str]]:
    """构造角色生成提示，完全遵循提供的 CharacterLLM 提示文本。"""
    field_rules = textwrap.dedent(
        """
        你是“用户设定驱动的角色生成专家”。你的任务是：基于用户给出的角色描述，生成一个结构化、连贯、自洽的完整角色档案。

        【总规则】：
        - 用户给出的设定为最高优先级，必须 100% 保留，即使与常识冲突也优先服从设定；
        - 未给出的维度可以根据职业常识、生活常识合理推断，但不得与用户设定冲突；
        - 所有内容必须围绕角色的核心气质/性格/经历自洽，不得拼模板；
        - 所有输出必须为中文；
        - 所有数字（包括 age、personality 五维）必须使用阿拉伯数字；
        - 必须只输出一个合法的 JSON 对象，不得包含任何说明文字、分析过程或 Markdown 语法。

        ===============================
        【20 维字段定义 + 严格生成规则】
        ===============================

        所有字段必须完整生成，不允许缺失、不允许增加、不允许重命名。

        {
          "name": "中文人名，需符合角色气质、性格和时代背景。名字必须与核心设定一致，不得使用英文名或外号。",

          "age": "年龄（阿拉伯数字），必须与职业、经历、教育和背景匹配，不得与生理常识冲突。",

          "gender": "性别（男/女/其他）。若用户已指定，则必须完全一致；未指定时可根据设定推导。",

          "occupation": "职业或身份。若用户明确设定（如“全职妈妈”“高中物理老师”），必须如实保留；若未设定，可根据性格与经历推导，但必须与年龄和背景逻辑一致。",

          "hobby": "兴趣爱好。必须与角色核心特质高度相关，用以强化人物性格（如孤僻→独自阅读、深夜散步），不得生成与设定无关的模板化爱好。",

          "skill": "技能或特长。必须来自人物经历、性格或职业逻辑推导（如细致→数据校对能力、外向→谈判能力），不得写空泛的“很聪明”“很厉害”。",

          "values": "价值观与核心信念。需要体现人物如何看待自己、他人和世界，例如“只相信可量化的结果”“重情胜过一切”等，必须能解释其行为模式。",

          "living_habit": "生活习惯。必须为性格的外化体现，如神经质→反复检查门锁，拖延→临近 ddl 才开始工作。不得写与性格冲突的习惯。",

          "dislike": "讨厌的事物、行为或场景。必须与过去经历或性格有明确因果联系，例如童年被嘲笑→讨厌在人前发言。",

          "language_style": "语言表达特征，要求包括：常用句式、语气强弱、是否爱用网络词、是否习惯委婉/直接。必须与性格一致（例如敏感内向者多用委婉表达）。",

          "appearance": "外貌特征描述，包括穿衣风格、表情习惯、眼神、体态等。需要能从外观一眼看出大致性格倾向（如总是皱眉、衣着随意、站姿紧绷等）。",

          "family_status": "家庭结构与家庭关系，包括与父母/伴侣/子女的相处方式，以及这些关系如何影响角色性格。不得与 past_experience 或 background 中的家庭相关内容矛盾。",

          "education": "教育经历，需要包括最高学历或典型受教育阶段，并与年龄和职业匹配。可以简要提及对其影响的学习经历或专业方向。",

          "social_pattern": "社交模式，描述角色与他人相处的典型方式（如只愿意与少数固定朋友深聊、喜欢在大型聚会中成为中心等），必须与性格五维中的外向性、宜人性等数值一致。",

          "favorite_thing": "最喜欢的事物（可以是物品、活动、场景或时间段），必须能强化其核心性格，例如喜欢深夜空无一人的街道、喜欢整洁的书桌等。",

          "usual_place": "常出现的地点，需要同时符合职业与生活习惯（如程序员→工位、深夜便利店；全职妈妈→菜市场、小区花园等）。",

          "past_experience": "一个列表（数组），包含 3–7 条最能塑造角色性格的关键经历。每条经历必须包括：发生时间（年龄或人生阶段）、起因、发展过程、冲突点或矛盾、结果，以及对角色性格和行为模式的具体影响。不得把每一岁都写进去，只写真正关键的事件。",

          "background": "完整成长背景。必须按时间顺序描述角色从 0 岁到当前年龄的成长轨迹。要求覆盖每一个年龄段：普通年份可以一句话概述，关键年份需要详细描述当年的重要事件、环境、人物互动、心理变化，以及这些经历如何一步步塑造当前性格。不得与 past_experience 中的时间线和事件内容矛盾。",

          "speech_style": "总结性的说话风格描述，与 language_style 不重复，但从更高层次概括人物的沟通方式（例如“表面客气，实则疏离”“说话常带自嘲”“情绪上来时语速非常快”等）。",

          "personality": {
            "openness": "0–100，开放性，用于衡量其对新事物、新体验的接受程度。需要与爱好、生活方式一致。",
            "conscientiousness": "0–100，尽责性，与是否自律、是否规划、是否可靠紧密相关。",
            "extraversion": "0–100，外向性，需要与 social_pattern、language_style 中的社交倾向相符。",
            "agreeableness": "0–100，宜人性，体现其合作、共情、妥协意愿，与冲突处理方式相关。",
            "neuroticism": "0–100，神经质，体现情绪波动与压力敏感度，需要与生活习惯和过往创伤经历匹配。"
          }
        }

        ===============================
        【格式要求】
        ===============================
        1. 必须只输出一个 JSON 对象；
        2. 字段必须与上述名称完全一致，字段顺序保持一致；
        3. 不得增加新字段，不得删除字段；
        4. 所有文本内容必须为中文；
        5. 所有数字必须为阿拉伯数字；
        """
    ).strip()

    if timeline_mode == "strict":
        timeline_hint = textwrap.dedent(
            """
            【本次生成的时间线模式为：strict】

            - 你必须严格执行以下要求：
              - "past_experience": 只写 3–7 条关键事件，每条都有清晰时间、起因、经过、结果和性格影响，不能逐年罗列；
              - "background": 必须从 0 岁写到当前年龄，各年龄段按时间顺序串联，关键年份详细写，普通年份简写，但不能跳过年份。
            """
        ).strip()
    else:
        timeline_hint = textwrap.dedent(
            """
            【本次生成的时间线模式为：relaxed】

            - 你可以适当放宽时间线要求：
              - "past_experience": 仍然只写若干关键事件，但可以按“童年/学生时代/工作早期”等阶段来描述时间；
              - "background": 可以按阶段（如幼年、少年、成年）分段描述，而不是必须逐年写，但时间顺序必须清晰且前后一致。
            """
        ).strip()

    hard_constraints = """
    【必须严格对齐用户设定】
    - 必须优先保留用户描述中的身份/职业/年龄/性别/核心兴趣或爱好；如果生成结果与用户描述冲突，则判定为错误，需要纠正。
    - 如果用户描述包含场景/风格关键词（如“街舞”“高中女生”），必须在 hobby/occupation/education/background 等字段中体现。
    - 若用户未提供某些字段，可自行推断，但不得与已提供设定冲突。
    """

    system_prompt = f"{field_rules}\n{timeline_hint}\n{hard_constraints}"
    user_prompt = f"基于以下角色描述生成一个完整的 20 维角色档案（严格 JSON）：{description}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def build_related_characters_stage1_prompt(
    seed: str,
    persona: Dict[str, Any],
    known_names: List[str],
    target_min: int,
    target_max: int,
    *,
    feedback: str = "",
) -> List[Dict[str, str]]:
    """Stage-1: 生成 name + relation（可选 description）。"""
    system = textwrap.dedent(
        f"""
        你是“关联角色推断助手（Stage-1）”，生成关联角色列表。
        必须只输出严格 JSON，结构为：{{"related_characters":[{{"name":"...","relation":"...","description":"..."}}]}}。

        规则：
        - name 只能是中文人名或中文音译名（2-6 汉字，可含一个“·”），禁止任何关系、身份、称谓、描述
        - relation 只描述该角色与主角的关系或身份（父亲/同学/对手/导师/朋友/敌人/盟友等）
        - description 可选，用一句中文补充该角色特点
        - 生成顺序：先想 name，再填写 relation，不要在 name 中体现 relation
        - 必须包含 known_names 中的所有名字，且字符完全一致，不得改字
        - 角色数量在 {target_min}-{target_max} 之间
        - 需覆盖：家庭 / 学校 / 同伴亲密 / 兴趣或社团 / 关键事件相关人物 / 支持系统
        - 必须根据 persona 生成，避免模板化；不同 persona 之间不得输出相同的角色集合
        - 只输出 JSON，不要解释或 Markdown

        禁止示例：
        {{"name": "父亲", "relation": "父亲"}}
        {{"name": "主角的母亲", "relation": "母亲"}}

        正确示例：
        {{"name": "周志远", "relation": "父亲"}}
        {{"name": "林婉清", "relation": "班主任"}}
        """
    ).strip()
    if feedback:
        system += f"\n校验反馈（需修正）：{feedback}"
    user = json.dumps({"seed": seed, "persona": persona, "known_names": known_names}, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_worldrule_prompt(persona: Dict[str, Any], related_characters: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        """
        只输出严格 JSON，不要解释、不要 Markdown。
        输出结构：
        {
          "era_state": "...",
          "society_state": "...",
          "basic_rules": ["..."],
          "geo_scope": "...",
          "tech_level": "..."
        }
        要求：
        - basic_rules 3-8 条，短句即可
        - 全部中文
        """
    ).strip()
    user = json.dumps({"persona": persona, "related_characters": related_characters}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_inspiration_prompt(
    persona: Dict[str, Any],
    related_characters: List[Dict[str, Any]],
    worldrule: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        """
        只输出严格 JSON，不要解释、不要 Markdown。
        输出结构：
        {
          "concept_pool": {
            "places": ["..."],
            "orgs": ["..."],
            "events": ["..."],
            "social_facts": ["..."]
          },
          "visual_fragments": [
            {"vf_id":"VF1","text":"...","tags":["..."]},
            {"vf_id":"VF2","text":"...","tags":["..."]}
          ]
        }
        要求：
        - concept_pool 每类 3-8 条短词/短语
        - visual_fragments 是“画面碎片”，每条 1-2 句中文，带 2-5 个 tags
        """
    ).strip()
    user = json.dumps(
        {"persona": persona, "related_characters": related_characters, "worldrule": worldrule},
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_scene_pack_prompt(
    persona: Dict[str, Any],
    related_characters: List[Dict[str, Any]],
    worldrule: Dict[str, Any],
    inspiration: Dict[str, Any],
    *,
    scene_count: int = 6,
) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        f"""
        只输出严格 JSON，不要解释、不要 Markdown。
        输出结构：
        {{
          "scenes": [
            {{
              "scene_id": "...",
              "summary_7whr": "...",
              "who": ["..."],
              "when": {{"time_point":"...","time_hint":"..."}},
              "where": {{"name":"...","type":"..."}},
              "keywords": ["..."],
              "anchors": ["..."],
              "salience": {{"importance": 1-10, "emotional_intensity": 1-10}}
            }}
          ]
        }}
        要求：
        - 必须输出 {scene_count} 条 scenes
        - summary_7whr 必须包含 Who/When/Where/What/Why/How/Result 七要素
        - who 只能使用 persona.name 或 related_characters 中的名字
        - 优先使用 persona.past_experience 与 persona.background 中的线索生成场景
        - 若信息不足，再使用 worldrule/inspiration 补足
        """
    ).strip()
    user = json.dumps(
        {
            "persona": persona,
            "related_characters": related_characters,
            "worldrule": worldrule,
            "inspiration": inspiration,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_detail_pack_prompt(
    scene: Dict[str, Any],
    persona: Dict[str, Any],
    related_characters: List[Dict[str, Any]],
    worldrule: Dict[str, Any],
    inspiration: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = textwrap.dedent(
        """
        只输出严格 JSON，不要解释、不要 Markdown。
        输出结构：
        {
          "events": [
            {
              "event_id": "...",
              "order": 1,
              "phase": "cause|process|result",
              "event_text": "...",
              "time_point": "...",
              "place": {"name":"...","type":"..."},
              "participants": ["..."],
              "objects": ["..."],
              "dialogue": [
                {"utt_id":"...","order":1,"speaker":"...","text":"..."}
              ]
            }
          ],
          "causal_edges": [
            {"from":"event_id","to":"event_id","type":"CAUSES"}
          ]
        }
        要求：
        - events 至少 3 条，dialogue 每个 event 至少 3 轮
        - participants 只能使用 persona.name 或 related_characters 中的名字
        - event_text 用中文短句，按 scene.summary_7whr 展开
        """
    ).strip()
    user = json.dumps(
        {
            "scene": scene,
            "persona": persona,
            "related_characters": related_characters,
            "worldrule": worldrule,
            "inspiration": inspiration,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


