import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

# Test Product ID from your previous log
PRODUCT_ID = "S5P_NRTI_L2__NO2____20260801T073526_20260801T074026_45597_03_020901_20260801T081223.nc"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_URL = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({PRODUCT_ID})/$value"

def test_cdse_access():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ Error: CDSE_CLIENT_ID or CDSE_CLIENT_SECRET missing in .env")
        return

    print("1. Fetching OAuth2 Access Token...")
    token_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }

    response = requests.post(TOKEN_URL, data=token_data)
    if response.status_code != 200:
        print(f"❌ Token fetch failed [{response.status_code}]: {response.text}")
        return

    access_token = response.json().get("access_token")
    print("✅ Token successfully retrieved!")

    print("\n2. Testing file download access endpoint (HTTP HEAD request)...")
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Using stream=True to avoid fetching the full file payload
    dl_response = requests.get(DOWNLOAD_URL, headers=headers, stream=True, allow_redirects=True)
    
    if dl_response.status_code == 200:
        print("✅ Success! Your credentials have full download access.")
    elif dl_response.status_code == 401:
        print("❌ 401 Unauthorized: Credentials generated, but download permission is blocked.")
        print("👉 Action required: Log into https://dataspace.copernicus.eu and accept the EULA/Terms of Use under your profile.")
    else:
        print(f"⚠️ Unexpected Status [{dl_response.status_code}]: {dl_response.reason}")

if __name__ == "__main__":
    test_cdse_access()