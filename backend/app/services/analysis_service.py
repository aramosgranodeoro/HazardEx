from langchain.messages import HumanMessage

from app import state


def build_initial_analysis_prompt(media_id: str, analysis_text: str) -> str:
    return f"""
        [Image attached, media_id={media_id}]
        These are the results of the initial automatic analysis of the multimedia content:
        {analysis_text}
        Write a clear and professional initial description of the content based solely on these results,
        without inventing or adding any additional information.
        Answer only in Spanish.
        """


async def register_media_in_agent_state(config: dict, media_id: str, media_type: str) -> None:
    await state.agent.aupdate_state(config, {"available_media": {media_id: media_type}})


async def run_initial_analysis(config: dict, media_id: str, analysis_text: str) -> str:
    """Registra el media en el estado del agente y obtiene la descripción inicial."""
    prompt = build_initial_analysis_prompt(media_id, analysis_text)
    response = await state.agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config)
    return response["messages"][-1].content