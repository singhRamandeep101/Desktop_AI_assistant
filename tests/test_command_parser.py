"""Tests for command parser."""

import pytest


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    class MockLogger:
        def log_info(self, msg):
            pass
        
        def log_error(self, msg, exc_info=False):
            pass
        
        def log_warning(self, msg):
            pass
        
        def log_performance(self, op, duration):
            pass
    
    return MockLogger()


def test_parse_greeting(mock_logger):
    """Test greeting command parsing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from command_parser import CommandParser
    
    parser = CommandParser(mock_logger)
    
    command = parser.parse_command("Hello Friday")
    assert command["intent"] == "greeting"
    assert "hello" in command["text"].lower()


def test_parse_time_query(mock_logger):
    """Test time query parsing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from command_parser import CommandParser
    
    parser = CommandParser(mock_logger)
    
    command = parser.parse_command("What time is it?")
    assert command["intent"] == "time_query"


def test_parse_exit(mock_logger):
    """Test exit command parsing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from command_parser import CommandParser
    
    parser = CommandParser(mock_logger)
    
    command = parser.parse_command("Exit")
    assert command["intent"] == "exit"


def test_parse_unknown(mock_logger):
    """Test unknown command parsing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from command_parser import CommandParser
    
    parser = CommandParser(mock_logger)
    
    command = parser.parse_command("Random text that doesn't match any pattern")
    assert command["intent"] in ["general", "unknown"]


def test_get_response(mock_logger):
    """Test response generation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from command_parser import CommandParser
    
    parser = CommandParser(mock_logger)
    
    command = {"intent": "greeting", "text": "Hello"}
    response = parser.get_response(command)
    assert len(response) > 0
    assert isinstance(response, str)


