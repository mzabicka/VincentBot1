import streamlit as st
import time
import uuid
import random
from datetime import datetime
from zoneinfo import ZoneInfo

# --- KONFIGURACJA ---
# Symulujemy wszystko, żeby nie wywalało błędów bibliotek
SHEET_ID = "DEMO"
SHEET_NAME = "Arkusz1"

# --- MOCKOWANIE (UDAVANIE) AI ---
# To pozwala działać aplikacji bez LangChain i OpenAI
class VincentSimulator:
    def invoke(self, inputs):
        user_text = inputs.get("input", "")
        # Zestaw odpowiedzi, które wyglądają naturalnie na screenach
        responses = [
            "Rozumiem, to brzmi jak trudne doświadczenie. Często mam wrażenie, że muszę być idealny, a to rodzi ogromną presję.",
            "Dziękuję, że się tym dzielisz. Zastanawiam się, co by się stało, gdybyś spróbował(a) spojrzeć na tę sytuację z większą łagodnością?",
            "To bardzo ludzkie podejście. Błędy są naturalną częścią procesu uczenia się, choć trudno je zaakceptować.",
            "Słyszę w Twoich słowach dużo emocji. Jak zazwyczaj radzisz sobie w takich momentach zwątpienia?",
            "Ciekawe. Czasami nasza wewnętrzna krytyka jest głośniejsza niż rzeczywistość. Czy myślisz, że to może być ten przypadek?"
        ]
        time.sleep(1.5) # Udajemy, że bot "myśli"
        return {"answer": random.choice(responses)}

# --- FUNKCJE POMOCNICZE ---
def save_to_sheets(data):
    # Udajemy zapis, żeby nie było błędu, jeśli klucze Google są złe
    print(f"Zapisano dane: {data}")

# --- STANY SESJI ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.group = "A" # Domyślna grupa do screenów
    st.session_state.chat_history = []
    st.session_state.page = "consent"
    st.session_state.rag_chain = VincentSimulator() # Podpinamy symulator
    st.session_state.start_time = None
    st.session_state.feedback_submitted = False
    st.session_state.demographics = {}
    st.session_state.pretest = {}

# --- EKRANY APLIKACJI ---

def consent_screen():
    st.title("Zaproszenie do udziału w badaniu")
    st.markdown("""
    Dziękuję za zainteresowanie moim badaniem!
    **Jestem studentką kierunku Psychologia i Informatyka na Uniwersytecie SWPS**.
    Badanie dotyczy interakcji z chatbotem. Trwa ok. 15-20 min.
    Jest anonimowe i dobrowolne.
    """)
    if st.checkbox("Wyrażam zgodę na udział w badaniu"):
        if st.button("Przejdź do badania"):
            st.session_state.page = "pretest"
            st.rerun()

def pretest_screen():
    st.title("Ankieta wstępna")
    st.subheader("Metryczka")
    st.number_input("Wiek (w latach)", 0, 99, None)
    st.selectbox("Płeć:", ["–– wybierz ––", "Kobieta", "Mężczyzna", "Inna", "Nie chcę podać"])
    st.selectbox("Wykształcenie:", ["–– wybierz ––", "Podstawowe", "Średnie", "Wyższe", "Inne"])
    
    st.subheader("Postawa wobec AI")
    items = [
        "Sztuczna inteligencja uczyni ten świat lepszym miejscem.",
        "Obawiam się sztucznej inteligencji.",
        "Chcę korzystać z technologii opartych na AI."
    ]
    for i, item in enumerate(items):
        st.radio(item, [1,2,3,4,5], index=None, key=f"ai_{i}", horizontal=True)
    
    st.subheader("Samopoczucie")
    st.slider("Suwak samopoczucia", 0, 100, 50, label_visibility="hidden")
    
    if st.button("Dalej"):
        st.session_state.page = "chat_instruction"
        st.rerun()

def chat_instruction_screen():
    st.title("Instrukcja")
    st.markdown("""
    Przed Tobą rozmowa z **Vincentem**.
    Rozmowa potrwa 10 minut.
    """)
    if st.button("Rozpocznij rozmowę"):
        st.session_state.page = "chat"
        st.rerun()

def chat_screen():
    st.title("Rozmowa z Vincentem")
    
    if not st.session_state.start_time:
        st.session_state.start_time = time.time()
    
    # Obliczanie czasu (dla screenów możemy to oszukać lub zostawić prawdziwy)
    elapsed = (time.time() - st.session_state.start_time) / 60
    
    # Wiadomość powitalna
    if not st.session_state.chat_history:
        welcome_msg = "Cześć, jestem Vincent. Dziś mam wrażenie, że nie jestem wystarczająco dobry. Jak Ty sobie radzisz, kiedy mimo wysiłku coś nie wychodzi?"
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
        
    # Wyświetlanie historii
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).markdown(msg["content"])
    
    # Obsługa czatu
    if user_input := st.chat_input("Napisz odpowiedź..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Vincent myśli..."):
            # Używamy symulatora - zero błędów OpenAI
            response = st.session_state.rag_chain.invoke({"input": user_input})
            reply = response["answer"]
            
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").markdown(reply)
            
    # Pasek postępu czasu
    remaining = max(0, 10 - elapsed)
    st.info(f"Do końca rozmowy pozostało ok. {int(remaining)} min.")
    
    # Przycisk awaryjny do przejścia dalej (widoczny po 10 min lub dla testów zawsze)
    if st.button("Zakończ rozmowę (Przejdź dalej)"):
        st.session_state.page = "posttest"
        st.rerun()

def posttest_screen():
    st.title("Ankieta końcowa")
    st.subheader("Samopoczucie")
    st.slider("Suwak samopoczucia", 0, 100, 50, key="post_vas", label_visibility="hidden")
    
    st.subheader("Samowspółczucie")
    items = [
        "Staram się być wyrozumiały dla swoich wad.",
        "Jestem krytyczny wobec siebie."
    ]
    for i, item in enumerate(items):
        st.radio(item, [1,2,3,4,5], index=None, key=f"scs_{i}", horizontal=True)
    
    st.subheader("Refleksja")
    st.text_area("O co chodziło w badaniu?")
    
    if st.button("Zakończ"):
        st.session_state.page = "thankyou"
        st.rerun()

def thankyou_screen():
    st.title("Dziękuję!")
    st.markdown("Twoje odpowiedzi zostały zapisane.")
    st.text_area("Co na plus?")
    st.text_area("Co na minus?")
    
    if st.button("Wyślij feedback", disabled=st.session_state.feedback_submitted):
        st.session_state.feedback_submitted = True
        st.success("Wysłano! Dziękuję za udział.")

# --- MAIN ---
def main():
    st.set_page_config(page_title="VincentBot", page_icon="🤖")
    
    if st.session_state.page == "consent": consent_screen()
    elif st.session_state.page == "pretest": pretest_screen()
    elif st.session_state.page == "chat_instruction": chat_instruction_screen()
    elif st.session_state.page == "chat": chat_screen()
    elif st.session_state.page == "posttest": posttest_screen()
    elif st.session_state.page == "thankyou": thankyou_screen()

if __name__ == "__main__":
    main()
