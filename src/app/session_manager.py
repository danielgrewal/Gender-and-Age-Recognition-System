import mysql.connector
from mysql.connector import Error
from PIL import Image

from app.image_manager import ImageManager

class SessionManager:
    
    def __init__(self):
        """ Initialize the SessionManager with database connection details. """
        
        self.host = "localhost"
        self.user = "root"
        self.password = "pass"
        self.database = "GARSDB"
        self.connection = None
    
    def connect(self):
        """ Connect to GARS Database """
        
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            
            return True
        
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close the database connection if it exists."""
        if self.connection:
            if self.connection.is_connected():
                self.connection.close()
                self.connection = None
    
    def execute_query(self, query, params = None):
        """ Execute a MySQL query. """
        
        # Return False if unable to connect
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if query.strip().upper().startswith("SELECT"):
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                self.connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                return affected_rows > 0
                
        except Error as e:
            print(f"Error executing query: {e}")
            return False
    
    def log_session(self, image: Image, age: int, gender: str, has_consent: bool):
        """ Store data if the user has provided consent """
        
        if not has_consent:
            return False
        
        # User provided consent to store image so serialize image for storage
        image_manager = ImageManager()
        image_encoded = image_manager.serialize(image)
        # Record image and prediction
        query = "insert into GARSDB.sessions (id, request_image, predicted_age, predicted_gender) values (NULL, %s, %s, %s)"
        params = (image_encoded, age, gender)
                  
        return self.execute_query(query, params)
    
    