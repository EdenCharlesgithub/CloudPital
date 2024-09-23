<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Dashboard</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <h1>Hospital Dashboard</h1>
    <input type="text" id="searchInput" placeholder="Search by Contact or Name">
    <div id="resultsContainer"></div>

    <script>
        $(document).ready(function() {
            $('#searchInput').on('input', function() {
                var searchTerm = $(this).val();
                if (searchTerm.length >= 2) {
                    $.ajax({
                        url: 'search.php',
                        type: 'POST',
                        data: {searchTerm: searchTerm},
                        success: function(response) {
                            $('#resultsContainer').html(response);
                        }
                    });
                }
            });
        });
    </script>
</body>
</html>
