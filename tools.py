from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import StructuredTool, Tool
from datetime import datetime

from pydantic import BaseModel, Field


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

write_to_file_tool = Tool.from_function(func=save_to_txt,
                                        name="Write_to_File",
                                        description="use this tool to save the research output to a text file"
                                        )

search = DuckDuckGoSearchRun()

def safe_duckduckgo(query: str):
    try:
        result = search.run(query)

        if not result:
            return "No search results found"

        return result

    except Exception as e:
        return f"DuckDuckGo failed: {str(e)}"

search_tool = Tool.from_function(
    func=safe_duckduckgo,
    name="DuckDuckGo_Search",
    description="Search the web for recent information"
)
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1000
)

def safe_wikipedia(query: str):
    try:
        result = api_wrapper.run(query)

        if not result:
            return "No Wikipedia result found"

        return result

    except Exception as e:
        return f"Wikipedia tool failed: {str(e)}"

wikipedia_tool = Tool.from_function(
    func=safe_wikipedia,
    name="Wikipedia",
    description="Search Wikipedia for general knowledge"
)


class TranslateInput(BaseModel):
    """
    Pydantic schema → each Field description becomes the per-argument
    hint the LLM reads when deciding what value to pass.
    """
    text: str = Field(
        description="The full text to be translated. Pass the exact content, not a summary."
    )
    target_language: str = Field(
        description=(
            "The language to translate into. Use the full English name, "
            "e.g. 'French', 'Arabic', 'Spanish', 'Japanese'."
        )
    )
    formality: str = Field(
        default="formal",
        description=(
            "Tone of the translation. Use 'formal' for academic/professional text, "
            "'informal' for casual/conversational text. Defaults to 'formal'."
        )
    )


def translate_text(text: str, target_language: str, formality: str = "formal") -> str:
    """
    Simulated translation (swap for DeepL / Google Translate API in production).
    The docstring here is the tool-level description; the Field descriptions
    above are the per-argument hints.
    """
    # In a real implementation you'd call an API here.
    return (
        f"[Simulated {formality} translation to {target_language}]\n"
        f"Original: {text[:80]}{'...' if len(text) > 80 else ''}\n"
        f"Translated: <{target_language} {formality} version of the text>"
    )


translate_tool = StructuredTool.from_function(
    func=translate_text,
    name="translate_text",
    description=(
        "Translate a piece of text into another language. "
        "Use when the user asks for a translation or when research sources are in a foreign language."
    ),
    args_schema=TranslateInput,    # ← this is what produces the rich JSON schema
)


'''



# ── NEW: Supabase / PostgreSQL tool ───────────────────────────────────────────

def _get_connection():
    """
    Reads credentials from environment variables.
    In Supabase: Settings → Database → Connection string (direct connection).
    Required env vars:
        SUPABASE_DB_HOST, SUPABASE_DB_PORT, SUPABASE_DB_NAME,
        SUPABASE_DB_USER, SUPABASE_DB_PASSWORD
    """
    return psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST"),
        port=os.getenv("SUPABASE_DB_PORT", "5432"),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        sslmode="require",          # Supabase always requires SSL
        connect_timeout=10,
    )


class DatabaseQueryInput(BaseModel):
    sql: str = Field(
        description=(
            "A read-only SQL SELECT statement to run against the research database. "
            "Do NOT use INSERT, UPDATE, DELETE, or DDL. "
            "Tables available: research_outputs(id, topic, summary, sources, created_at)."
        )
    )
    query_description: str = Field(
        description=(
            "A plain-English explanation of what this query retrieves and why. "
            "Used for logging and audit purposes."
        )
    )


def query_database(sql: str, query_description: str) -> str:
    """
    Executes a read-only SQL query against the Supabase PostgreSQL database.
    Always SELECT only — mutations are blocked as a safety measure.
    """
    # Safety guard: block any non-SELECT statement
    normalized = sql.strip().upper()
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")
    if not normalized.startswith("SELECT"):
        return "Error: Only SELECT statements are permitted."
    if any(normalized.startswith(kw) or f" {kw} " in normalized for kw in forbidden):
        return "Error: Mutation statements are not allowed."

    print(f"\n[DB Tool] {query_description}\n[DB Tool] SQL: {sql}\n")  # audit log

    try:
        conn = _get_connection()
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(50)   # cap at 50 rows to stay within context limits
                if not rows:
                    return "Query returned no rows."
                # Convert to readable string for the LLM
                lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
                return f"Query returned {len(rows)} row(s):\n" + "\n".join(lines)
    except psycopg2.Error as e:
        return f"Database error: {e.pgerror or str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()


db_query_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description=(
        "Query the Supabase PostgreSQL research database using a SELECT statement. "
        "Use this to look up previously saved research topics, summaries, or sources."
    ),
    args_schema=DatabaseQueryInput,
)

'''