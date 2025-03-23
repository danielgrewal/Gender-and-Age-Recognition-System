import pytest
from PIL import Image
from app.session_manager import SessionManager
from app.image_manager import ImageManager


@pytest.fixture
def session_manager():
    manager = SessionManager()
    manager.connect()
    yield manager
    manager.disconnect()

def test_log_session_writes_to_db(session_manager):
    # Dummy data
    image = Image.new("RGB", (10, 10), color="blue")
    age = -1
    gender = "Male"
    has_consent = True

    result = session_manager.log_session(image, age, gender, has_consent)

    # Check that the write was successful
    assert result is True 

    # Retrieve written records from DB 
    records = session_manager.execute_query("SELECT * FROM GARSDB.sessions WHERE predicted_age = -1")
    
    # Check if record was returned
    assert records is not None

    # Check if data was written successfully
    record = records[0]
    assert record["predicted_age"] == age
    assert record["predicted_gender"] == gender
    assert record["request_image"] is not None
    
    # use ImageManager to deserialize image
    img_manager = ImageManager()
    result = img_manager.deserialize(record["request_image"])
    
    # Check result
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'
    assert result.size == (10, 10)

    # Clean-up inserted dummy data
    session_manager.execute_query("DELETE FROM GARSDB.sessions WHERE predicted_age = -1")

def test_log_session_does_not_write_to_db_with_no_consent(session_manager):
    
    # Dummy data
    image = Image.new("RGB", (10, 10), color="blue")
    age = -2
    gender = "Female"
    has_consent = False

    result = session_manager.log_session(image, age, gender, has_consent)

    # Check that the write was successful
    assert result is False 

    # Retrieve any potentially written records from DB 
    records = session_manager.execute_query("SELECT * FROM GARSDB.sessions WHERE predicted_age = -2")
    
    # Ensure no record was returned
    assert len(records) == 0