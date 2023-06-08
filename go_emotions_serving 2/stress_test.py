import concurrent.futures
import requests
import time

# Set up number of requests you want to send
NUM_REQUESTS = 3

# Define the URL and payload
url = "http://localhost/predict"
payload = {
    "text": "Man is like hell why does he cry"
}

headers = {
    'Content-Type': 'application/json'
}


def send_request(url, payload, headers):
    # Send a single request
    response = requests.request("POST", url, headers=headers, json=payload)
    return response.status_code


# Record start time
start_time = time.time()

# Send requests concurrently and record responses
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(send_request, url, payload, headers)
               for _ in range(NUM_REQUESTS)}

responses = [future.result()
             for future in concurrent.futures.as_completed(futures)]

# Record end time
end_time = time.time()

# Calculate and print statistics
total_requests = len(responses)
# Assuming 200 is the HTTP status code for a successful request
successful_requests = responses.count(200)
failed_requests = total_requests - successful_requests
time_taken = end_time - start_time
req_per_second = total_requests / time_taken

print(f"Total requests: {total_requests}")
print(f"Successful requests: {successful_requests}")
print(f"Failed requests: {failed_requests}")
print(f"Time taken: {time_taken} seconds")
print(f"Requests per second: {req_per_second}")
