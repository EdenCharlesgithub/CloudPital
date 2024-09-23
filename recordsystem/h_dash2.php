<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Details</title>
</head>
<body>
    <?php
    include('conn.php');

    if ($_SERVER['REQUEST_METHOD'] == 'GET' && isset($_GET['userId'])) {
        $userId = mysqli_real_escape_string($conn, $_GET['userId']);
        $query = "SELECT * FROM users WHERE id = $userId";
        $result = mysqli_query($conn, $query);

        if (mysqli_num_rows($result) > 0) {
            $row = mysqli_fetch_assoc($result);

            echo '<h1>User Details</h1>';
            echo '<p>Name: ' . $row['name'] . '</p>';
            echo '<p>Contact: ' . $row['contact'] . '</p>';
            echo '<p>Age: ' . $row['age'] . '</p>';
            echo '<p>Gender: ' . $row['gender'] . '</p>';

            // Add Edit/Update options here...
        } else {
            echo '<p>User not found.</p>';
        }
    } else {
        echo '<p>Invalid request.</p>';
    }
    ?>
</body>
</html>
