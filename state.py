# Enhanced state.py with context support
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Existing state variables
authenticated_users = {}
user_sessions = {} 
processed_messages = set()

# New context management
user_contexts = {}

def cleanup_old_contexts():
    """Clean up contexts older than 1 hour to prevent memory leaks."""
    current_time = datetime.now()
    contexts_to_remove = []
    
    for account_number, context in user_contexts.items():
        if context.get('timestamp') and (current_time - context['timestamp']).seconds > 3600:
            contexts_to_remove.append(account_number)
    
    for account_number in contexts_to_remove:
        del user_contexts[account_number]
        logger.info(f"Cleaned up old context for account: {account_number}")

def cleanup_old_processed_messages():
    """Clean up old processed messages to prevent memory leaks."""
    # Keep only last 1000 message IDs
    if len(processed_messages) > 1000:
        # Convert to list, keep last 1000, convert back to set
        recent_messages = list(processed_messages)[-1000:]
        processed_messages.clear()
        processed_messages.update(recent_messages)
        logger.info("Cleaned up old processed messages")

# Periodic cleanup (you might want to call this periodically in your webhook)
def periodic_cleanup():
    """Run periodic cleanup of old data."""
    cleanup_old_contexts()
    cleanup_old_processed_messages()