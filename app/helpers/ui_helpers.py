import streamlit as st
from helpers.chatbot_helpers import get_bot_reply


def floating_chatbot():

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("Bot", "👋 Hi! I’m the StockVision Assistant. Ask me about stocks, models, or predictions.")
        ]

    # Floating CSS
    st.markdown("""
    <style>
    .chat-float {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 10000;
    }

    .chat-float button {
        background-color: #0d6efd !important;
        color: white !important;
        border-radius: 50%;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown('<div class="chat-float">', unsafe_allow_html=True)

    with st.popover("💬", use_container_width=False):
        st.markdown("### 🤖 StockVision Assistant")

        # Show messages
        for sender, msg in st.session_state.chat_history[-10:]:
            st.markdown(f"**{sender}:** {msg}")

        # 🚀 FAST INPUT USING FORM
        with st.form(key="chat_form", clear_on_submit=True):
            user_msg = st.text_input(
                "Type your question",
                placeholder="Ask about project, models, ARIMA, LSTM..."
            )
            submitted = st.form_submit_button("Send")

        if submitted and user_msg:
            reply = get_bot_reply(user_msg)
            st.session_state.chat_history.append(("You", user_msg))
            st.session_state.chat_history.append(("Bot", reply))

        col1, col2 = st.columns(2)
        if col1.button("🧹 Clear Chat"):
            st.session_state.chat_history = [
                ("Bot", "👋 Chat cleared! Ask me something new.")
            ]

    st.markdown('</div>', unsafe_allow_html=True)
