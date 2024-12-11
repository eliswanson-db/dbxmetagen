def exponential_backoff(retries, base_delay=1, max_delay=120, jitter=True):
    """
    Exponential backoff with optional jitter.
    
    :param retries: Number of retries attempted.
    :param base_delay: Initial delay in seconds.
    :param max_delay: Maximum delay in seconds.
    :param jitter: Whether to add random jitter to the delay.
    :return: None
    """
    delay = min(base_delay * (2 ** retries), max_delay)
    if jitter:
        delay = delay / 2 + random.uniform(0, delay / 2)
    time.sleep(delay)