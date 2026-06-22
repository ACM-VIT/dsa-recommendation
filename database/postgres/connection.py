import os
import psycopg2

DATABASE_URL = os.environ.get("postgresql://neondb_owner:npg_iG0vSgMk1Qyj@ep-falling-darkness-aqwf6czy-pooler.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

def get_connection():
    return psycopg2.connect(DATABASE_URL)