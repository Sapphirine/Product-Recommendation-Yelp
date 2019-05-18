#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  data <- reactive({
    x <- df
  })
  
  output$mymap <- renderLeaflet({
    df <- data()
    
    m <- leaflet(data = df) %>%
      addTiles() %>%
      setView(lng=-115, lat=36 , zoom=10) %>%
      addMarkers(lng = ~longitude,
                 lat = ~latitude,
                 popup = paste("name", df$name, "<br>",
                               "star:", df$est_mean))
    m
  })
})
