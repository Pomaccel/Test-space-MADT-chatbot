import streamlit as st
import google.generativeai as genai

# Initialize session state variables if not already present
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None 

if "greeted" not in st.session_state:
    st.session_state.greeted = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input_history" not in st.session_state:
    st.session_state.user_input_history = []

if "rerun_needed" not in st.session_state:
    st.session_state.rerun_needed = False  # Flag to control reruns

# Sidebar to display user input history as buttons
st.sidebar.title("User Input History")

# Add a dropdown for selecting options
dropdown_option = st.sidebar.selectbox(
    "Select an option:",
    ["0.Overview", "1.Predict", "2.Sale By location"]
)

# Add "Home" and "Clear History" buttons in the sidebar
if st.sidebar.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.user_input_history = []
    st.session_state.greeted = False
    st.session_state.rerun_needed = True  # Set flag to trigger a rerun

# Loop through the user input history and create a button for each one
for i, prompt in enumerate(st.session_state.user_input_history, start=1):
    if st.sidebar.button(f"{i}. {prompt}"):
        # Update chat history and re-run the selected conversation
        st.session_state.chat_history = [("user", prompt)]  # Start fresh with that prompt        
        st.session_state.rerun_needed = True  # Set flag to trigger a rerun

        # Now handle the conversation logic below after rerun
        user_input = prompt
        try:
            # Prompt to generate a response from the model
            query_prompt = f"""You are an AI assistant that transforms user questions into SQL queries to retrieve data from a BigQuery database. 
            Use the schema information and generate a SQL query based on the user's input. User's input: '{user_input}'."""

            # Simulate a call to the model
            response = model.generate_content(query_prompt)
            bot_response = response.text

            # Append AI response to chat history
            st.session_state.chat_history.append(("assistant", bot_response))

        except Exception as e:
            st.error(f"Error generating AI response: {e}")
        break  # Exit the loop after processing the first clicked history button

# Main application title
st.title("LLM to SQL command")

# Input for Gemini API Key
gemini_api_key = st.text_input("Gemini API Key: ", placeholder="Type your API Key here...", type="password")

# Configure Gemini API
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-pro")
    except Exception as e:
        st.error(f"Error configuring Gemini API Key: {e}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # Generate greeting if not already greeted
    if not st.session_state.greeted:
        greeting_prompt = "Greet the user as a friendly and knowledgeable data engineer. \
                        Introduce yourself (you are AI assistant) and let the user know you're here to assist with \
                        any questions they may have about transforming user questions into SQL queries to retrieve data from a BigQuery database."

        try:
            response = model.generate_content(greeting_prompt)
            bot_response = response.text
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
            st.session_state.greeted = True
        except Exception as e:
            st.error(f"Error generating AI greeting: {e}")

    # Input box for user's message
    if user_input := st.chat_input("Type your message here..."):
        # Append user input to chat history and user input history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.user_input_history.append(user_input)
        st.chat_message("user").markdown(user_input)

        try:
            prompt = """You are an AI assistant that transforms user questions into SQL queries to retrieve data from a BigQuery database. 
                    Below is the detailed schema of the database, including table names, column names, data types, and descriptions. 
                    Use this information to generate accurate SQL queries based on user input. 
                    ### Data Dictionary 

                    #### Table: customer 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | CustomerID       | STRING      |                                             | 
                    | Customer         | STRING      |                                             | 
                    | Country          | STRING      | Country                                     | 
                    | OpticMainID      | STRING      |                                             | 
                    | Category         | STRING      |                                             | 
                    | zoneId           | STRING      |                                             |

                    #### Table: date_dimension 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | date_key         | STRING      |                                             | 
                    | full_date        | DATETIME    |                                             | 
                    | year             | INT64       |                                             | 
                    | month            | INT64       |                                             | 
                    | day              | INT64       |                                             | 
                    | week             | INT64       |                                             |  
                    | quarter          | INT64       |                                             |
                    
                    #### Table: invoice 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | InvoiceNo        | STRING      |                                             | 
                    | CustomerID       | STRING      |                                             | 
                    | InvoiceDate      | DATE        |                                             | 

                    #### Table: invoice_type 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | typeId           | STRING      |                                             | 
                    | type_name        | STRING      |                                             | 

                    #### Table: product
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | ProductId        | STRING      |                                             | 
                    | lentype          | STRING      |                                             |
                    | Part_Description | STRING      |                                             | 
                    | Material_Type    | STRING      |                                             | 
                    | Lens_Type        | STRING      |                                             | 
                    | price            | FLOAT64     |                                             | 

                    #### Table: reorder 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | reorder_cause_id | STRING      |                                             | 
                    | cause            | STRING      |                                             |

                    #### Table: store 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | StoreId          | STRING      |                                             | 
                    | Store            | STRING      |                                             | 
                    
                    #### Table: Transaction 
                    | Column Name       | Data Type   | Description                                 | 
                    |-------------------|-------------|---------------------------------------------| 
                    | InvoiceNo         | STRING      |                                             | 
                    | ProductId         | STRING      |                                             | 
                    | Quantity          | STRING      |                                             | 
                    | StoreId           | INT64       |                                             | 
                    | TypeId            | STRING      |                                             | 
                    | Reordered_Cause_ID| STRING      |                                             | 

                    #### Table: zoning 
                    | Column Name      | Data Type   | Description                                 | 
                    |------------------|-------------|---------------------------------------------| 
                    | ZoneId           | STRING      |                                             | 
                    | provinceEN       | STRING      |                                             | 
                    | provinceTH       | STRING      |                                             | 
                    | Region           | STRING      |                                             | 
                    
                    ### Relationships between tables: 
                    - `invoice.CustomerID` references `customer.CustomerID`.
                    - `transaction.InvoiceNo` references `invoice.InvoiceNo`.
                    - `transaction.ProductId` references `product.ProductId`.
                    - `transaction.StoreId` references `store.StoreId`.
                    - `transaction.TypeId` references `invoice_type.typeId`.
                    - `transaction.Reordered_Cause_ID` references `reorder.reorder_cause_id`.
                    - `customer.zoneId` references `zoning.ZoneId`.
                    - `invoice.InvoiceDate` references `date_dimension.full_date`. \n
                    """

            # Add chat history to the prompt
            for role, message in st.session_state.chat_history:
                prompt += f"{role}: {message}\n"

            prompt += f"User: {user_input}\n"
            prompt += """Based on the user's input, generate an appropriate SQL query to retrieve the requested information from the database. 
                        Be precise and ensure that the query follows SQL syntax correctly."""

            response = model.generate_content(prompt)
            bot_response = response.text

            # Append AI response to chat history
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)

        except Exception as e:
            st.error(f"Error generating AI response: {e}")

# Check if rerun is needed
if st.session_state.rerun_needed:
    st.session_state.rerun_needed = False  # Reset flag
    st.rerun()  # This will trigger a refresh
