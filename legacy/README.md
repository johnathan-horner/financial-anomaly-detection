# Legacy prototype (not used)

Code from before the October 2026 rebuild: a FastAPI service, AWS CDK stacks, Lambda and ECS
handlers, LangGraph and Bedrock agent code, a PyTorch autoencoder that was never trained, and an MCP
server whose tools returned random numbers. The Streamlit app showed mock data.

Nothing in the current code imports from here. It is kept so no history is lost on the rebuild
branch and can be removed once that is approved. The same code is on `main` and on the tag
`backup/pre-rebuild-2026-10-23`.
