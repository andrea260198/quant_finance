# NOTE: remember to start the Docker engine before

# Create Docker img
docker build -t test_img .

# Run Docker container
docker run test_img

# Run Docker container with user inputs
# docker run -t -i test_img
