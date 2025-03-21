from threading import Lock

# Dictionary to hold shared runtime state
state = {
    "current_element": None,
    "current_state": None,
    # Add additional shared keys as needed
}

# A lock to ensure thread-safe updates
state_lock = Lock()
