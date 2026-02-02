from pymongo import MongoClient
from bson.objectid import ObjectId
from models import User, Company

# --- Database Configuration ---
MONGO_DETAILS = "mongodb+srv://infocyberbells:URgCpmEAgksetiiI@cyberbellsmongocluster.vy8xm.mongodb.net/resumes?retryWrites=true&w=majority&serverSelectionTimeoutMS=5000&connectTimeoutMS=5000"
client = MongoClient(MONGO_DETAILS)
database = client.get_database()
user_collection = database.get_collection("users")
company_collection = database.get_collection("companies")

# --- Helper Functions ---

def user_helper(user) -> dict:
    """
    Converts a user document from MongoDB to a dictionary.
    """
    return {
        "id": str(user["_id"]),
        "username": user.get("username"),
        "email": user.get("email"),
        "user_type": user.get("user_type"),
        "company_id": user.get("company_id"),
    }

def company_helper(company) -> dict:
    """
    Converts a company document from MongoDB to a dictionary.
    """
    return {
        "id": str(company["_id"]),
        "company_name": company.get("company_name"),
        "email": company.get("email"),
        "phone_no": company.get("phone_no"),
        "staff": company.get("staff"),
    }

# --- Company CRUD ---

def add_company(company_data: dict) -> dict:
    """
    Adds a new company to the database.
    """
    company = company_collection.insert_one(company_data)
    new_company = company_collection.find_one({"_id": company.inserted_id})
    return company_helper(new_company)

def retrieve_company_by_email(email: str) -> dict:
    """
    Retrieves a company by email.
    """
    company = company_collection.find_one({"email": email})
    if company:
        return company
    return None

def retrieve_company_by_name(company_name: str) -> dict:
    """
    Retrieves a company by its name.
    """
    company = company_collection.find_one({"company_name": company_name})
    if company:
        return company
    return None

# --- User (Recruiter/Seeker) CRUD ---

def add_user(user_data: dict) -> dict:
    """
    Adds a new user (recruiter or seeker) to the database.
    """
    user = user_collection.insert_one(user_data)
    new_user = user_collection.find_one({"_id": user.inserted_id})
    return user_helper(new_user)

def retrieve_user_by_email(email: str) -> dict:
    """
    Retrieves a user by email.
    """
    user = user_collection.find_one({"email": email})
    if user:
        return user
    return None
