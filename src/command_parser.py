import time
from typing import Any, Dict


class CommandParser:
    def __init__(self, logger):
        self.logger = logger
        self.intent_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings"],
            "time_query": ["time", "what time", "current time", "clock"],
            "date_query": ["date", "what date", "today", "what day"],
            "weather_query": ["weather", "temperature", "forecast", "rain"],
            "exit": ["stop", "exit", "quit", "goodbye", "bye"],
            "help": ["help", "what can you do", "commands"],
        }
    
    def parse_command(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"intent": "unknown", "text": ""}
        
        text_lower = text.lower().strip()
        start_time = time.time()
        
        intent = "unknown"
        action = None
        parameters = {}
        confidence = 0.0
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    intent = intent_name
                    confidence = 0.8
                    break
            if intent != "unknown":
                break
        
        if intent == "unknown":
            intent = "general"
            confidence = 0.5
        
        if intent == "weather_query":
            location = self._extract_location(text_lower)
            if location:
                parameters["location"] = location
        
        parse_time = time.time() - start_time
        self.logger.log_performance("parse_command", parse_time)
        
        return {
            "intent": intent,
            "action": action,
            "text": text,
            "parameters": parameters,
            "confidence": confidence
        }
    
    def _extract_location(self, text: str) -> str:
        location_keywords = ["in", "at", "for"]
        words = text.split()
        
        for i, word in enumerate(words):
            if word in location_keywords and i + 1 < len(words):
                return " ".join(words[i + 1:])
        
        return ""
    
    def get_response(self, command: Dict[str, Any]) -> str:
        intent = command.get("intent", "unknown")
        
        responses = {
            "greeting": "Hello! How can I help you?",
            "time_query": self._get_time_response(),
            "date_query": self._get_date_response(),
            "weather_query": "I'm sorry, weather information is not available yet.",
            "help": self._get_help_response(),
            "exit": "Goodbye!",
            "general": f"I heard: {command.get('text', '')}",
            "unknown": "I'm sorry, I didn't understand that."
        }
        
        return responses.get(intent, responses["unknown"])
    
    def _get_time_response(self) -> str:
        import datetime
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"
    
    def _get_date_response(self) -> str:
        import datetime
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        return f"Today is {current_date}"
    
    def _get_help_response(self) -> str:
        return (
            "I can help you with:\n"
            "- Telling the time\n"
            "- Telling the date\n"
            "- General conversation\n"
            "Say 'exit' to quit."
        )


