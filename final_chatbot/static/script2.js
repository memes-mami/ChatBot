function checkEmail() {
    var emailFormat = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    var email = document.getElementById("email").value.trim();
    if (email.match(emailFormat) && email.endsWith('@student.nitandhra.ac.in')) {
        window.location.href = "basetemp2.html";
    } else {
        alert("Invalid email format or domain. Please enter a valid *@student.nitandhra.ac.in email.");
    }
}
