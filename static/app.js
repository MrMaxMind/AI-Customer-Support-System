$(document).ready(function() {
    $('#generateForm').on('submit', function(event) {
        event.preventDefault();
        const userInput = $('#userInput').val();

        // Disable the submit button to prevent multiple clicks
        $('button[type="submit"]').prop('disabled', true);

        $.ajax({
            url: '/generate',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ user_input: userInput }),
            success: function(response) {
                // Display the generated text
                $('#output').html('<p>' + response.generated_text + '</p>');
                $('#output').show();

                // Enable the submit button for future queries
                $('button[type="submit"]').prop('disabled', false);
                // $('#userInput').val(''); // Clear the input field
            },
            error: function() {
                $('#output').html('<p class="text-danger">An error occurred. Please try again.</p>');
                $('#output').show();

                // Enable the submit button again in case of an error
                $('button[type="submit"]').prop('disabled', false);
            }
        });
    });
});

