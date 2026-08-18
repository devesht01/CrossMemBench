def split_system_and_user(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user
