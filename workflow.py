import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    print("1. Starting research workflow...")
    workflow = ResearchWorkflow(
        research_objective="Role of BRCA1 in cancer development",
        artifacts_dir="./artifacts"
    )
    #result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    print("2. Running research workflow...")
    result = await workflow.run(num_cycles=1, tasks_per_cycle=1)
    print("3. Generating report...")
    report = await workflow.generate_report()
    print("4. done...")
    print(report)

print("5. -----.")

asyncio.run(run())
print("6. ------")
