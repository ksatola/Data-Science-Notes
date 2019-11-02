
install.packages("shiny") # once per machine 
library("shiny") # in each relevant script

# The UI is the result of calling the `fluidPage()` layout function
my_ui <- fluidPage(
    # A static content element: a 2nd level header that shows "Greetings from Shiny"
    h2("Greetings from Shiny"),
    
    # A widget: a text input box (save input in the `username` key)
    textInput(inputId = "username", label = "What is your name?"),
    
    # An output element: a text output (for the `message` key)
    textOutput(outputId = "message") 
)

# The server is a function that takes `input` and `output` arguments
my_server <- function(input, output) {
    # Assign a value to the `message` key in the `output` list
    # using Shiny's renderText() method, creating a value the UI can display
    output$message <- renderText({
        # This block is like a function that will automatically rerun # when a referenced `input` value changes
        # Use the `username` key from input to create a value
        message_str <- paste0("Hello ", input$username, "!")
        # Return the value to be rendered by the UI
        message_str })
}

# To start running your app, pass the variables defined in previous code snippets
# into the `shinyApp()` function
shinyApp(ui = my_ui, server = my_server)
