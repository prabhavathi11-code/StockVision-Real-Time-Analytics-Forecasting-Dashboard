function validateUser() {
    let u = document.getElementById("username").value.trim();
    let p = document.getElementById("password").value.trim();

    const users = {
        "B.Kusuma Priya": "@4203",
        "Ch.Prabhavathi": "@4211",
        "A.Teja": "@2303"
    };

    if (users[u] && users[u] === p) {
        alert("Login Successful!");

        // Redirect to Streamlit App
        window.location.href = "http://localhost:8501";

    } else {
        alert("Invalid Credentials!");
    }
}
