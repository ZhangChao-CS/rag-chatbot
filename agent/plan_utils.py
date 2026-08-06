from agent.planning.schema import Plan, Task


def format_prior_outputs(task: Task, plan: Plan) -> str:
    deps = plan.get_dependency_results(task)
    if not deps:
        return "无"
    lines = []
    for dep in deps:
        if dep.result:
            lines.append(
                f"[Task #{dep.id}] {dep.description}\n  output: {dep.result.output}"
            )
    return "\n".join(lines) if lines else "无"
