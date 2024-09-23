<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Patient Register</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <link rel="stylesheet" href="../css/style.css">

  </head>
  <body>
<!--PHP Code-->
<?php
include 'conn.php';

if (isset($_SESSION['contact'])) {
  header("Location: p_dash.php"); // Redirect to p_dash.php
  exit();
}

// Check if the request method is POST
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Get the input values from the form
    $contact = $_POST['contact'];
    $name = $_POST['name'];
    $password = $_POST['password'];

    // Sanitize input to prevent SQL injection
    $contact = mysqli_real_escape_string($conn, $contact);
    $name = mysqli_real_escape_string($conn, $name);
    $password = mysqli_real_escape_string($conn, $password);

    // Check if the contact number already exists
    $checkQuery = "SELECT * FROM users WHERE contact='$contact'";
    $result = mysqli_query($conn, $checkQuery);

    if (mysqli_num_rows($result) > 0) {
        // User already exists
        echo '<script>alert("User with this contact number already exists.Try Login");</script>';
    } else {
        // Insert new user
        $insertQuery = "INSERT INTO users (contact, name, password) VALUES ('$contact', '$name', '$password')";
        if (mysqli_query($conn, $insertQuery)) {
          echo '<script>alert("User registered successfully! Login Now");</script>';

        } else {
          echo '<script>alert("Error: ' . mysqli_error($conn) . '");</script>';
        }
    }

    // Close the connection
    mysqli_close($conn);
}
?>


  <nav class="navbar bg-body-tertiary" style="background: rgba( 255, 255, 255, 0.25 );
          box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
          backdrop-filter: blur( 4px );
          -webkit-backdrop-filter: blur( 4px );
          border-radius: 10px;
          border: 1px solid rgba( 255, 255, 255, 0.18 );">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <img src="https://i.ibb.co/4RVT7kB/acube-traders-1.png" alt="Bootstrap" width="40" height="40">
                    CloudPital
                  </a>
              <form class="d-flex" role="search">
                <a href="http://127.0.0.1:5000/"><button type="button" class="btn btn-success">CheckUp</button></a>
                <a href="logout.php"><button type="button" class="btn btn-danger">Logout</button></a>
              </form>
            </div>
          </nav>

          <div class="login-page">
          <h1 style="text-align:center;">Patient Register</h1>
  <div class="form">
    <form class="login-form" action="p_reg.php" method="POST">
      <input type="number" name="contact" placeholder="contact number" />
      <input type="text" name="name" placeholder="full name" />
      <input type="password" name="password" placeholder="password"/>
      <button type="submit" value="Register">create</button>
      <p class="message">Already registered? <a href="./p_login.php">Login Here</a></p>
    </form>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
<script>
  $('.message a').click(function(){
   $('form').animate({height: "toggle", opacity: "toggle"}, "slow");
});
</script>
  </body>
</html>



