import streamlit as st
import sys
import os
import time
import uuid
import random
from datetime import datetime
from zoneinfo import ZoneInfo

# --- IMPORTY BIBLIOTEK (PEŁNA WERSJA AI) ---
import openai
import gspread
from google.oauth2.service_account import Credentials

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# --- KONFIGURACJA ---
SHEET_ID = "1LnCkrWY271w2z3VSMAVaKqqr7U4hqGppDTVuHvT5sdc"
SHEET_NAME = "Arkusz1"
# Używamy nowej nazwy folderu, aby uniknąć konfliktów ze starymi plikami
FAISS_INDEX_PATH = "./faiss_db_2025" 

# --- USTAWIENIE KLUCZA API (BEZPIECZNE) ---
# Ustawiamy jako zmienną środowiskową, co zapobiega błędom walidacji Pydantic
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
except Exception:
    # Jeśli uruchamiasz lokalnie bez secrets.toml, to może być puste, ale na Streamlit Cloud zadziała
    pass

@st.cache_resource(show_spinner=False)
def get_sheet():
    try:
        creds_info = {
            "type": st.secrets["GDRIVE_TYPE"],
            "project_id": st.secrets["GDRIVE_PROJECT_ID"],
            "private_key_id": st.secrets["GDRIVE_PRIVATE_KEY_ID"],
            "private_key": st.secrets["GDRIVE_PRIVATE_KEY"],
            "client_email": st.secrets["GDRIVE_CLIENT_EMAIL"],
            "client_id": st.secrets["GDRIVE_CLIENT_ID"],
            "auth_uri": st.secrets["GDRIVE_AUTH_URI"],
            "token_uri": st.secrets["GDRIVE_TOKEN_URI"],
            "auth_provider_x509_cert_url": st.secrets["GDRIVE_AUTH_PROVIDER_CERT_URL"],
            "client_x509_cert_url": st.secrets["GDRIVE_CLIENT_CERT_URL"]
        }
        
        _gspread_creds = Credentials.from_service_account_info(
            creds_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ])
        
        _gspread_client = gspread.authorize(_gspread_creds)
        sheet = _gspread_client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
        return sheet 
    except Exception as e:
        print(f"Błąd połączenia z Arkuszem Google: {e}")
        return None

PDF_FILE_PATHS = [
    "docs/The Mindful Self-Compassion Workbook A Proven Way to Accept Yourself, Build Inner Strength, and Thrive.pdf",
    "docs/Self-Compassion The Proven Power of Being Kind to Yourself.pdf"
]

self_compassion_items = [
    "Staram się być wyrozumiały i cierpliwy w stosunku do tych aspektów mojej osoby, których nie lubię.",
    "Kiedy przechodzę przez bardzo trudny okres, staram się być łagodny i troskliwy w stosunku do siebie.",
    "Kiedy coś mnie denerwuje, staram się zachować równowagę emocjonalną.",
    "Kiedy czuję się jakoś gorsza/gorszy, staram się pamiętać, że większość ludzi tak ma.",
    "Jestem krytyczny i mało wyrozumiały wobec moich własnych wad i niedociągnięć.",
    "Jestem nietolerancyjny i niecierpliwy wobec tych aspektów mojej osoby, których nie lubię."]
  
ai_attitude_items = {
    "Sztuczna inteligencja uczyni ten świat lepszym miejscem.": "ai_1",
    "Mam silne negatywne emocje w stosunku do sztucznej inteligencji.": "ai_2",
    "Chcę korzystać z technologii opartych na sztucznej inteligencji.": "ai_3",
    "Sztuczna inteligencja ma więcej wad niż zalet.": "ai_4",
    "Z niecierpliwością czekam na przyszły rozwój sztucznej inteligencji.": "ai_5",
    "Sztuczna inteligencja oferuje rozwiązania wielu światowych problemów.": "ai_6",
    "Preferuję technologie, które nie zawierają sztucznej inteligencji.": "ai_7",
    "Obawiam się sztucznej inteligencji.": "ai_8",
    "Wybrałbym/wybrałabym raczej technologię ze sztuczną inteligencją niż bez niej.": "ai_9",
    "Sztuczna inteligencja raczej tworzy problemy niż je rozwiązuje.": "ai_10",
    "Kiedy myślę o sztucznej inteligencji, mam głównie pozytywne odczucia.": "ai_11",
    "Raczej unikałbym/unikałabym technologii opartych na sztucznej inteligencji.": "ai_12"
}

def save_to_sheets(data_dict):
    try:
        sheet = get_sheet()
        if not sheet: return
        
        user_id = data_dict.get("user_id")
        if not user_id: return

        current_headers = sheet.row_values(1) 
        all_potential_headers = list(current_headers)
        for key in data_dict.keys():
            if key not in all_potential_headers: all_potential_headers.append(key)
        
        if not current_headers:
            sheet.insert_row(all_potential_headers, 1)
            current_headers = all_potential_headers 
        else:
            headers_to_add = [h for h in all_potential_headers if h not in current_headers]
            if headers_to_add:
                all_records = sheet.get_all_records() 
                sheet.clear() 
                sheet.insert_row(all_potential_headers, 1) 
                if all_records:
                    rows_to_insert = []
                    for record in all_records:
                        row = [str(record.get(h, "")) for h in all_potential_headers]
                        rows_to_insert.append(row)
                    sheet.append_rows(rows_to_insert)
                current_headers = all_potential_headers 

        user_ids_in_sheet = []
        user_id_col_index = -1
        if "user_id" in current_headers:
            user_id_col_index = current_headers.index("user_id") + 1
            user_ids_in_sheet = sheet.col_values(user_id_col_index)[1:] 

        row_index = -1
        if user_id_col_index != -1 and user_id in user_ids_in_sheet:
            row_index = user_ids_in_sheet.index(user_id) + 2 
        
        if row_index != -1:
            existing_row_values = sheet.row_values(row_index)
            existing_data_map = {}
            for i, header in enumerate(current_headers):
                if i < len(existing_row_values): existing_data_map[header] = existing_row_values[i]
                else: existing_data_map[header] = "" 
            merged_data = {**existing_data_map, **data_dict}
            row_to_update = [str(merged_data.get(header, "")) for header in current_headers]
            # Używamy nazwanych argumentów, aby uniknąć DeprecationWarning
            sheet.update(range_name=f"A{row_index}", values=[row_to_update])
        else:
            new_row_values = [str(data_dict.get(header, "")) for header in current_headers]
            sheet.append_row(new_row_values)
    except Exception as e:
        print(f"Błąd zapisu danych: {e}")

@st.cache_resource(show_spinner=False)
def load_resources(PDF_FILE_PATHS):
    # 1. Inicjalizacja modelu embeddingów
    embedding_model = HuggingFaceEmbeddings(
        model_name='all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = None
    
    # 2. Próba wczytania istniejącej bazy
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Nie udało się wczytać starej bazy: {e}. Tworzę nową.")
            vector_store = None # Wymuszamy stworzenie nowej

    # 3. Jeśli bazy nie ma (lub była zepsuta), tworzymy nową z PDFów
    if vector_store is None:
        docs = []
        for path in PDF_FILE_PATHS:
            if os.path.exists(path):
                try:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Błąd odczytu PDF {path}: {e}")
        
        if docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vector_store = FAISS.from_documents(documents=splits, embedding=embedding_model)
            # Możemy spróbować zapisać, ale na Streamlit Cloud to i tak zniknie po restarcie
            # vector_store.save_local(FAISS_INDEX_PATH)
        else:
            # Fallback, jeśli nie ma PDFów, żeby aplikacja nie padła
            vector_store = FAISS.from_texts(["To jest pusta baza wiedzy."], embedding=embedding_model)

    # 4. Inicjalizacja LLM (Chatu)
    # WAŻNE: Nie przekazujemy tu 'openai_api_key' ani 'proxies', 
    # ponieważ ustalamy to w zmiennych środowiskowych na górze pliku.
    chat = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.0
    )
    return vector_store, chat

def setup_rag_system(PDF_FILE_PATHS, user_gender_instruction):
    vector_store, chat = load_resources(PDF_FILE_PATHS)

    history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Biorąc pod uwagę powyższą rozmowę, wygeneruj zapytanie do wyszukania w bazie wiedzy, aby odpowiedzieć na ostatnie pytanie."),
    ])

    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(chat, retriever, history_aware_retriever_prompt)

    vincent_identity_instruction = (
        "Masz własną tożsamość: jesteś **Vincent**, cyfrową istotą używającą **formy męskiej**. "
        "W swoich wypowiedziach mówisz o sobie w rodzaju męskim. "
    )

    user_gender_instruction = st.session_state.get("gender_instruction", "")
    gender_instruction = f"{vincent_identity_instruction}\n{user_gender_instruction}"

    SYSTEM_PROMPT = f"""
        {gender_instruction}
        Jesteś Vincent — cyfrową istotą. Starasz się zrozumieć ludzkie podejście do trudności.
        Twoje wypowiedzi mają 2–4 zdania. W rozmowie bądź empatyczny.
        Korzystaj z dostarczonego kontekstu (Context), aby inspirować się wiedzą o samowspółczuciu, ale nie cytuj go wprost jak encyklopedii.
        **Język:** wyłącznie polski.
        """

    MASTER_PROMPT = """\
        <context>
        {context}
        </context>
        Użytkownik napisał: "{input}"
        """

    Youtubeing_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", MASTER_PROMPT),
    ])

    document_chain = create_stuff_documents_chain(chat, Youtubeing_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    return retrieval_chain

# --- STANY APLIKACJI ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.group = None
    st.session_state.chat_history = []
    st.session_state.shuffled_pretest_items = {}
    st.session_state.shuffled_posttest_items = {}
    st.session_state.page = "consent"
    st.session_state.rag_chain = None

# --- EKRANY ---

def consent_screen():
    st.title("Zaproszenie do udziału w badaniu")
    st.markdown("""
    Dziękuję za zainteresowanie moim badaniem!
    **Jestem studentką kierunku Psychologia i Informatyka na Uniwersytecie SWPS**.
    Badanie dotyczy interakcji z chatbotem. Trwa ok. 15-20 min.
    Jest anonimowe i dobrowolne.
    """)
    if st.checkbox("Wyrażam zgodę na udział w badaniu"):
        if st.button("Przejdź do badania", key="go_pre"):
            now = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
            sheet = get_sheet()
            try: 
                existing = sheet.col_values(sheet.row_values(1).index("group")+1)[1:]
                st.session_state.group = "A" if len(existing) % 2 == 0 else "B"
            except: st.session_state.group = "A"
            
            st.session_state.timestamp_start_initial = now
            save_to_sheets({"user_id": st.session_state.user_id, "group": st.session_state.group, "timestamp_start": now, "status": "start"})
            st.session_state.page = "pretest"
            st.rerun()

def pretest_screen():
    st.title("Ankieta wstępna")
    st.subheader("Metryczka")
    age = st.number_input("Wiek (w latach)", 0, 99, None)
    gender = st.selectbox("Płeć:", ["–– wybierz ––", "Kobieta", "Mężczyzna", "Inna", "Nie chcę podać"])
    edu = st.selectbox("Wykształcenie:", ["–– wybierz ––", "Podstawowe", "Gimnazjalne", "Zasadnicze zawodowe", "Średnie", "Pomaturalne", "Wyższe licencjackie/inżynierskie", "Wyższe magisterskie", "Doktoranckie lub wyższe", "Inne", "Nie chcę podać"])
    
    st.subheader("Postawa wobec AI")
    atts = {}
    for i, k in ai_attitude_items.items(): atts[k] = st.radio(i, [1,2,3,4,5], index=None, horizontal=True)
    
    st.subheader("Samopoczucie")
    vas = st.slider("Suwak samopoczucia", 0, 100, 50, label_visibility="hidden")
    
    if st.button("Dalej", key="go_chat"):
        if age and gender!="–– wybierz ––" and edu!="–– wybierz ––" and all(v for v in atts.values()):
            st.session_state.demographics = {"age": age, "gender": gender, "education": edu}
            st.session_state.pretest = {"wellbeing_vas": vas, "ai_attitude": atts}
            now = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.pretest_timestamp = now
            
            data = {"user_id": st.session_state.user_id, "timestamp_pretest_end": now, "status": "pretest_ok"}
            data.update({f"demographics_{k}":v for k,v in st.session_state.demographics.items()})
            data.update({f"pre_ai_{k}":v for k,v in atts.items()})
            data["pre_wellbeing_vas"] = vas
            save_to_sheets(data)

            if gender == "Kobieta": st.session_state.gender_instruction = "Użytkownik to kobieta. Zwracaj się w formie żeńskiej."
            elif gender == "Mężczyzna": st.session_state.gender_instruction = "Użytkownik to mężczyzna. Zwracaj się w formie męskiej."
            else: st.session_state.gender_instruction = "Użytkownik neutralny. Unikaj form męskich/żeńskich."
            
            st.session_state.page = "chat_instruction"
            st.rerun()
        else: st.warning("Proszę wypełnić wszystkie pola.")

def chat_instruction_screen():
    st.title("Instrukcja")
    st.markdown("Przed Tobą rozmowa z Vincentem. Potrwa 10 minut.")
    if st.button("Rozpocznij rozmowę"):
        st.session_state.page = "chat"
        st.rerun()

def chat_screen():
    st.title("Rozmowa z Vincentem")
    if st.session_state.rag_chain is None:
        with st.spinner("Ładowanie Vincenta... (To może chwilę potrwać przy pierwszym uruchomieniu)"):
            st.session_state.rag_chain = setup_rag_system(PDF_FILE_PATHS, st.session_state.gender_instruction)
    
    if not st.session_state.start_time: st.session_state.start_time = time.time()
    elapsed = (time.time() - st.session_state.start_time) / 60
    
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "assistant", "content": "Cześć, jestem Vincent. Dziś mam wrażenie, że nie jestem wystarczająco dobry. Jak Ty sobie radzisz, kiedy mimo wysiłku coś nie wychodzi?"})
        
    for msg in st.session_state.chat_history: st.chat_message(msg["role"]).markdown(msg["content"])
    
    if user_input := st.chat_input("Napisz odpowiedź..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Vincent myśli..."):
            try:
                hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.chat_history[-6:]]
                if hist and isinstance(hist[-1], HumanMessage): hist.pop()
                resp = st.session_state.rag_chain.invoke({"input": user_input, "chat_history": hist})["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.chat_message("assistant").markdown(resp)
            except Exception as e: 
                st.error(f"Wystąpił błąd: {e}")
            
    if elapsed >= 10:
        if st.button("Zakończ rozmowę"):
            now = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_timestamp = now
            log = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            save_to_sheets({"user_id": st.session_state.user_id, "timestamp_chat_end": now, "status": "chat_ok", "conversation_log": log})
            st.session_state.page = "posttest"
            st.rerun()
    else: st.info(f"Pozostało ok. {int(11-elapsed)} min.")

def posttest_screen():
    st.title("Ankieta końcowa")
    st.subheader("Samopoczucie")
    vas = st.slider("Suwak samopoczucia", 0, 100, 50, label_visibility="hidden")
    
    st.subheader("Samowspółczucie")
    scs = {}
    for i, item in enumerate(self_compassion_items): scs[f"SCS_{i+1}"] = st.radio(item, [1,2,3,4,5], index=None, horizontal=True)
    
    st.subheader("Refleksja")
    ref = st.text_area("O co chodziło w badaniu?")
    
    if st.button("Zakończ"):
        if all(v for v in scs.values()):
            now = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
            data = {"user_id": st.session_state.user_id, "timestamp_posttest_end": now, "status": "done", "post_wellbeing_vas": vas, "post_reflection": ref}
            data.update({f"post_{k}":v for k,v in scs.items()})
            
            # Zapisz wszystko jeszcze raz dla pewności
            data.update({f"demographics_{k}":v for k,v in st.session_state.demographics.items()})
            data.update({f"pre_ai_{k}":v for k,v in st.session_state.pretest.get("ai_attitude", {}).items()})
            data["pre_wellbeing_vas"] = st.session_state.pretest.get("wellbeing_vas")
            
            save_to_sheets(data)
            st.session_state.page = "thankyou"
            st.rerun()
        else: st.warning("Proszę wypełnić wszystkie pola.")

def thankyou_screen():
    st.title("Dziękuję!")
    st.markdown("Twoje odpowiedzi zostały zapisane.")
    pos = st.text_area("Co na plus?")
    neg = st.text_area("Co na minus?")
    if st.button("Wyślij feedback", disabled=st.session_state.feedback_submitted):
        now = datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
        save_to_sheets({"user_id": st.session_state.user_id, "timestamp_feedback": now, "feedback_pos": pos, "feedback_neg": neg})
        st.session_state.feedback_submitted = True
        st.success("Wysłano!")

def main():
    st.set_page_config(page_title="VincentBot", page_icon="🤖")
    if "page" not in st.session_state: 
        st.session_state.page = "consent"
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.start_time = None
        st.session_state.feedback_submitted = False
        st.session_state.demographics = {}
        st.session_state.pretest = {}

    if st.session_state.page == "consent": consent_screen()
    elif st.session_state.page == "pretest": pretest_screen()
    elif st.session_state.page == "chat_instruction": chat_instruction_screen()
    elif st.session_state.page == "chat": chat_screen()
    elif st.session_state.page == "posttest": posttest_screen()
    elif st.session_state.page == "thankyou": thankyou_screen()

if __name__ == "__main__":
    main()
