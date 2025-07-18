"""
Quotex API Credentials Management
Secure handling of API keys and authentication
"""

import os
from typing import Optional
from cryptography.fernet import Fernet
import base64
import hashlib

class CredentialsManager:
    def __init__(self):
        self.encryption_key = self._get_or_create_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key for credentials"""
        key_file = "config/.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def encrypt_credential(self, credential: str) -> str:
        """Encrypt a credential string"""
        return self.fernet.encrypt(credential.encode()).decode()
    
    def decrypt_credential(self, encrypted_credential: str) -> str:
        """Decrypt a credential string"""
        return self.fernet.decrypt(encrypted_credential.encode()).decode()
    
    def save_credentials(self, email: str, password: str, api_key: Optional[str] = None):
        """Save encrypted credentials to file"""
        creds = {
            'email': self.encrypt_credential(email),
            'password': self.encrypt_credential(password),
            'api_key': self.encrypt_credential(api_key) if api_key else None
        }
        
        with open('config/.credentials', 'w') as f:
            import json
            json.dump(creds, f)
    
    def load_credentials(self) -> dict:
        """Load and decrypt credentials from file"""
        try:
            with open('config/.credentials', 'r') as f:
                import json
                encrypted_creds = json.load(f)
            
            return {
                'email': self.decrypt_credential(encrypted_creds['email']),
                'password': self.decrypt_credential(encrypted_creds['password']),
                'api_key': self.decrypt_credential(encrypted_creds['api_key']) if encrypted_creds.get('api_key') else None
            }
        except FileNotFoundError:
            return {}

# Quotex API Configuration
QUOTEX_CONFIG = {
    'base_url': 'https://qxbroker.com',
    'api_url': 'https://qxbroker.com/api',
    'websocket_url': 'wss://ws.qxbroker.com/socket.io/',
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1
}

# Demo vs Live trading
TRADING_MODE = {
    'demo': True,  # Set to False for live trading
    'demo_balance': 10000,
    'live_balance': 0
}

# Initialize credentials manager
credentials_manager = CredentialsManager()

def get_quotex_credentials() -> dict:
    """Get Quotex credentials from environment or file"""
    # Try environment variables first
    email = os.getenv('QUOTEX_EMAIL')
    password = os.getenv('QUOTEX_PASSWORD')
    api_key = os.getenv('QUOTEX_API_KEY')
    
    if email and password:
        return {
            'email': email,
            'password': password,
            'api_key': api_key
        }
    
    # Try encrypted file
    return credentials_manager.load_credentials()

def setup_credentials():
    """Interactive setup for credentials"""
    print("Setting up Quotex credentials...")
    email = input("Enter your Quotex email: ")
    password = input("Enter your Quotex password: ")
    api_key = input("Enter your Quotex API key (optional): ")
    
    credentials_manager.save_credentials(email, password, api_key)
    print("Credentials saved securely!")

if __name__ == "__main__":
    setup_credentials()