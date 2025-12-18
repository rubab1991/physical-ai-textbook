from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key for JWT - in production, this should be in environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Security scheme for bearer token
security = HTTPBearer()

# Request/Response models
class UserRegistration(BaseModel):
    email: str
    name: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Mock user database (in real implementation, this would be a proper database)
users_db = {}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(email: str, password: str):
    user = users_db.get(email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """
    Verify JWT token and return user data
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")

        # Check if token is expired (using iat claim)
        token_iat = payload.get("iat")
        if token_iat:
            issued_at = datetime.utcfromtimestamp(token_iat)
            if datetime.utcnow() > issued_at + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES * 2):  # 2x for grace period
                raise HTTPException(status_code=401, detail="Token has expired")

        return email
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current user from token
    """
    token = credentials.credentials
    email = verify_token(token)

    user = users_db.get(email)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "created_at": user["created_at"]
    }

@router.post("/auth/register", response_model=UserResponse)
async def register_user(user: UserRegistration):
    """
    Register a new user with email, name, and password
    """
    # Check if user already exists
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Validate email format
    if "@" not in user.email or "." not in user.email:
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Validate password strength
    if len(user.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")

    # Hash the password
    hashed_password = get_password_hash(user.password)

    # Create user
    user_id = str(uuid.uuid4())
    new_user = {
        "id": user_id,
        "email": user.email,
        "name": user.name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
        "preferences": {
            "difficulty_level": "intermediate",
            "learning_style": "balanced",
            "language_preference": "en",
            "content_focus": "conceptual"
        },
        "is_active": True
    }

    users_db[user.email] = new_user

    return UserResponse(
        id=new_user["id"],
        email=new_user["email"],
        name=new_user["name"],
        created_at=new_user["created_at"]
    )

@router.post("/auth/login", response_model=Token)
async def login_user(user_credentials: UserLogin):
    """
    Authenticate user and return access token
    """
    user = authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/auth/logout")
async def logout_user(request: Request):
    """
    Logout user (currently just a placeholder - in real implementation
    this would involve token blacklisting)
    """
    # In a real implementation, this would add the token to a blacklist
    # For now, we'll just return a success message
    return {"message": "Successfully logged out"}

@router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        created_at=current_user["created_at"]
    )

@router.put("/auth/me")
async def update_current_user_info(
    updated_user: UserRegistration,
    current_user: dict = Depends(get_current_user)
):
    """
    Update current user information
    """
    user_email = current_user["email"]
    user_record = users_db[user_email]

    # Update user info
    user_record["name"] = updated_user.name

    # If password is being updated
    if updated_user.password:
        if len(updated_user.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
        user_record["hashed_password"] = get_password_hash(updated_user.password)

    # Update timestamp
    user_record["updated_at"] = datetime.utcnow()

    return UserResponse(
        id=user_record["id"],
        email=user_record["email"],
        name=user_record["name"],
        created_at=user_record["created_at"]
    )