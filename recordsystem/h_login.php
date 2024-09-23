<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hospital Login</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <link rel="stylesheet" href="../css/style.css">

  </head>
  <body>

<?php
session_start();
include 'conn.php';

if (isset($_SESSION['name'])) {
  header("Location: h_dash.php"); // Redirect to h_dash.php
  exit();
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {

$name = $_POST['name'];
$password = $_POST['password'];

// Check if user exists based on name number and password
$sql = "SELECT * FROM hospital WHERE name='$name' AND password='$password'";
$result = mysqli_query($conn, $sql);

if (mysqli_num_rows($result) == 1) {
    $_SESSION['name'] = $name;
    header('Location: h_dash.php');
} else {
    echo "Invalid name number or password.";
}
}
mysqli_close($conn);
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
          <h1 style="text-align:center;">Hospital Login</h1>
  <div class="form">
    <form class="login-form" action="h_login.php" method="POST">
      <input type="name" name="name" placeholder="name"/>
      <input type="password" name="password" placeholder="password"/>
      <button>login</button>
      <p class="message">Not registered? <a href="./h_reg.php">Create an account</a></p>
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