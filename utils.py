from huggingface_hub import login

# ============================================================================
# HUGGINGFACE SETUP
# ============================================================================

def huggingface_login(token: str):
    """
    Login in HuggingFace Hub for model access.
    
    Args:
        token: HuggingFace Hub token
    """
    if token is not None:
        login(token=token)
    else:
        print("No HUGGINGFACE_HUB_TOKEN found. Make sure you're logged in via 'huggingface-cli login'")
