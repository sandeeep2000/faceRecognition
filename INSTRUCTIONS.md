# Face Recognition Application Setup Instructions

This guide provides instructions for setting up and running the Face Recognition application on a new Windows or macOS system. This application uses the `face_recognition` library.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Python**: Version 3.8 or newer.
    *   Download from python.org.
    *   During installation (especially on Windows), ensure "Add Python to PATH" is checked.
*   **pip**: Usually comes with Python. You can upgrade it using `python -m pip install --upgrade pip`.
*   **Git**: For cloning the repository (if you get this from Git).
    *   Download from git-scm.com.
*   **PostgreSQL**: A PostgreSQL database server.
    *   Download from postgresql.org/download.
    *   During setup, you'll typically create a default superuser (often `postgres`) and set a password. Remember these credentials.
*   **C++ Compiler and Build Tools**: Required for `dlib`, a core dependency of `face_recognition`.
    *   **Windows**:
        *   Install **Microsoft C++ Build Tools**. Go to the Visual Studio Downloads page, scroll down to "Tools for Visual Studio", and download "Build Tools for Visual Studio".
        *   Run the installer and select the "Desktop development with C++" workload.
    *   **macOS**:
        *   Install **Xcode Command Line Tools** by running the following command in your terminal:
            ```bash
            xcode-select --install
            ```
*   **CMake**: Also required for `dlib`.
    *   **Windows**:
        *   The easiest way is often to install it via pip *before* `dlib` or `face_recognition`:
            ```bash
            pip install cmake
            ```
        *   Alternatively, download an installer from cmake.org/download and ensure it's added to your system's PATH.
    *   **macOS**:
        *   Using Homebrew (recommended):
            ```bash
            brew install cmake
            ```
        *   Alternatively, download from cmake.org/download.

## 2. Project Setup

1.  **Get the Code**:
    *   If cloned from Git:
        ```bash
        git clone <repository_url>
        cd <project_directory_name>
        ```
    *   If you have the files directly, navigate to the project's root directory (e.g., `MyCleanFaceApp`).

2.  **Create and Activate a Virtual Environment**:
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   **Windows (Command Prompt)**: `venv\Scripts\activate`
    *   **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
        (If PowerShell errors, run: `Set-ExecutionPolicy Unrestricted -Scope Process`)
    *   **macOS/Linux**: `source venv/bin/activate`
    Your terminal prompt should now start with `(venv)`.

3.  **Install Dependencies**:
    Ensure you have a `requirements.txt` file in the project root. The content should be similar to:
    ```txt
    Flask
    Flask-SQLAlchemy
    Flask-Marshmallow
    psycopg2-binary
    face_recognition
    opencv-python
    Pillow
    numpy
    python-dotenv
    # Add any other specific dependencies your project uses
    ```
    Install them:
    ```bash
    pip install -r requirements.txt
    ```
    **Important Note on `face_recognition` / `dlib` Installation**:
    *   This can be challenging. Ensure C++ compiler and CMake are correctly installed (see Prerequisites).
    *   If issues persist, consult the `face_recognition` library's installation guide: https://github.com/ageitgey/face_recognition#installation

## 3. Database Configuration

1.  **Create PostgreSQL Database**:
    *   Connect to your PostgreSQL server (e.g., using `psql` or pgAdmin).
    *   Create a new database. Example: `face_recognition_db`.
        ```sql
        CREATE DATABASE face_recognition_db;
        ```

2.  **Set Environment Variables**:
    Create a file named `.env` in the root of your project directory. **Do not commit this file to public Git repositories if it contains sensitive credentials.**

    **`.env` file content (example):**
    ```env
    # Adjust with your actual PostgreSQL credentials and database name
    DATABASE_URL=postgresql://postgres:your_password@localhost:5432/face_recognition_db

    # Generate a strong, random secret key for Flask
    # You can generate one using Python: python -c "import os; print(os.urandom(24).hex())"
    SECRET_KEY=your_very_strong_random_secret_key_here
    ```
    *   Replace `your_password` and `your_very_strong_random_secret_key_here` accordingly.

## 4. Running the Application

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project's root directory** in your terminal.
3.  **Run the Flask application**:
    ```bash
    python app.py
    ```
    The application should start. On the first run, it will create database tables.
    You should see output like `* Running on http://0.0.0.0:5000/`.

4.  **Access the Application**:
    Open your web browser and go to `http://localhost:5000`.

## 5. Troubleshooting Common Issues

*   **`dlib` or `face_recognition` installation errors**:
    *   Double-check C++ compiler and CMake installation and PATH.
    *   On Windows, try installing from a "Developer Command Prompt for VS".
    *   Clean pip cache: `pip cache purge`.
*   **Database Connection Errors (`OperationalError`, etc.)**:
    *   Verify PostgreSQL server is running.
    *   Check `DATABASE_URL` in `.env` is correct.
    *   Ensure PostgreSQL user has permissions.
*   **"ModuleNotFoundError"**:
    *   Ensure virtual environment is activated.
    *   Confirm all dependencies from `requirements.txt` installed correctly.
*   **Webcam Issues (`cv2.VideoCapture(0)` fails)**:
    *   Ensure webcam is connected and not used by another app.
    *   Try different camera indices (e.g., `1`, `2`).
    *   On Windows, `cv2.VideoCapture(0, cv2.CAP_DSHOW)` (used in `app.py`) is often more reliable.

## 6. Development Tips

*   **Flask Debug Mode**: `app.run(debug=True)` enables auto-reloading and an in-browser debugger.
*   **`use_reloader=False`**: Set in `app.py` for stability with webcam resources during development. For production, `debug` should be `False`.

