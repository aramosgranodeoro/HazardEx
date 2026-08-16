from langchain.messages import AIMessage, HumanMessage
from app.agent.graph import build_agent_graph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from app.agent.graph import build_agent_graph

agent_builder = build_agent_graph()
agent = agent_builder.compile(checkpointer=MemorySaver())

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# --- Hilo 1: conversación con contexto encadenado ---
config_1 = {"configurable": {"thread_id": "test-1"}}
agent.update_state(
    config_1,
    {"messages": [AIMessage(content="Resultado del análisis: fuego detectado en la esquina superior derecha de la imagen, nivel de confianza 0.89, humo visible en zona central.")]}
)

print("=" * 60)
print("TURNO 1 (thread test-1)")
print("=" * 60)
result = agent.invoke(
    {"messages": [HumanMessage(content="¿Dónde se encuentra el fuego en la imagen?")]},
    config=config_1
)
for m in result["messages"]:
    m.pretty_print()

print("=" * 60)
print("TURNO 2 (thread test-1) — pregunta de seguimiento, sin repetir contexto")
print("=" * 60)
result = agent.invoke(
    {"messages": [HumanMessage(content="¿Y qué nivel de riesgo le darías?")]},
    config=config_1
)
for m in result["messages"]:
    m.pretty_print()

print("=" * 60)
print("TURNO 3 (thread test-1) — referencia implícita al turno 1")
print("=" * 60)
result = agent.invoke(
    {"messages": [HumanMessage(content="¿Puedes recordarme qué te pregunté al principio?")]},
    config=config_1
)
for m in result["messages"]:
    m.pretty_print()

# --- Hilo 2: conversación completamente distinta, para comprobar aislamiento ---
config_2 = {"configurable": {"thread_id": "test-2"}}

print("=" * 60)
print("TURNO 1 (thread test-2) — hilo nuevo, no debería saber nada del anterior")
print("=" * 60)
result = agent.invoke(
    {"messages": [HumanMessage(content="¿Qué te pregunté antes en esta conversación?")]},
    config=config_2
)
for m in result["messages"]:
    m.pretty_print()

# --- Verificación directa del estado guardado en el checkpointer ---
print("=" * 60)
print("ESTADO COMPLETO GUARDADO para thread test-1")
print("=" * 60)
snapshot = agent.get_state(config_1)
for m in snapshot.values["messages"]:
    m.pretty_print()

print(f"\nNúmero de llamadas al LLM en test-1: {snapshot.values.get('llm_calls')}")