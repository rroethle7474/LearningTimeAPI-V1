def format_duration(duration: str) -> str:
    """Convert ISO 8601 duration to readable format"""
    print("DURATION", duration)
    if not duration or not isinstance(duration, str):
        return ""
    
    # Check if already formatted (contains ':')
    if ':' in duration:
        return duration
        
    # Check if it's an ISO duration (starts with 'PT')
    if not duration.startswith('PT'):
        return duration
    
    # Remove 'PT' prefix
    duration = duration[2:]
    
    # Initialize variables
    hours = 0
    minutes = 0
    seconds = 0
    
    # Find hours
    if 'H' in duration:
        hours, duration = duration.split('H')
        hours = int(hours)
    
    # Find minutes
    if 'M' in duration:
        minutes, duration = duration.split('M')
        minutes = int(minutes)
    
    # Find seconds
    if 'S' in duration:
        seconds = int(duration.rstrip('S'))
    
    # Format the output
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}" 