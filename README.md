# Video Resizer

A Streamlit application for video resizing using AWS S3.

## Features

- **Video Resizer:** Upload and resize videos for various social media platforms.

## Setup Instructions

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/bbassetttubi/video_resizer.git
    cd video_resizer
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory of the project and add the following:

    ```env
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    ```

5. **Run the application:**

    ```bash
    streamlit run app.py
    ```

## Usage

Navigate to the local server address provided by Streamlit (usually `http://localhost:8501`) in your web browser. From there, you can access the video resizing feature.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
