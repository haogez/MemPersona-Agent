from mem_persona_agent.persona.schema import Persona


def test_persona_schema_fields():
    data = {
        "name": "测试角色",
        "age": 28,
        "gender": "女",
        "occupation": "工程师",
        "hobby": "阅读",
        "skill": "编程",
        "values": "真诚",
        "living_habit": "早睡早起",
        "dislike": "谎言",
        "language_style": "温和",
        "appearance": "清秀",
        "family_status": "独生女",
        "education": "本科",
        "social_pattern": "内向",
        "favorite_thing": "猫",
        "usual_place": "图书馆",
        "past_experience": ["童年在南方成长"],
        "background": "普通家庭",
        "speech_style": "有条理",
        "personality": {
            "openness": 70,
            "conscientiousness": 60,
            "extraversion": 40,
            "agreeableness": 80,
            "neuroticism": 30,
        },
    }
    persona = Persona.model_validate(data)
    assert persona.name == "测试角色"
    assert persona.personality.openness == 70
