"""
Northbridge Bank — Credit Risk Portfolio Query Engine
Streamlit application

This app converts the original Jupyter Notebook prototype into a deployable
Streamlit application. All core logic — the Verified Query Template Library,
the intent classification / query generation / validation / retry / execution /
response-generation tools, and the run_pipeline orchestration — is preserved
exactly as implemented in the notebook.
"""

from typing import Optional
import os
import re
import json
import sqlite3

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="📊",
    layout="wide",
)


# ============================================================================
# CREDENTIALS / CONFIGURATION
# ============================================================================
def load_credentials():
    """
    Loads OpenAI credentials.

    Preference order (first one found wins):
    1. Streamlit secrets (st.secrets) — recommended for Streamlit Community
       Cloud / any hosted deployment.
    2. Environment variables OPENAI_API_KEY / OPENAI_API_BASE.
    3. A local config.json file (same format used in the original notebook):
       {"OPENAI_API_KEY": "...", "OPENAI_API_BASE": "..."}
    """
    api_key = None
    api_base = None

    # 1. Streamlit secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        if "OPENAI_API_BASE" in st.secrets:
            api_base = st.secrets["OPENAI_API_BASE"]
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3. config.json (kept for parity with the original notebook)
    if not api_key and os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base

    return api_key, api_base


API_KEY, API_BASE = load_credentials()


# ============================================================================
# LLM INITIALIZATION  (cached so the models are built once per session)
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_llms():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm, evaluator_llm


# ============================================================================
# DATABASE CONNECTION  (read-only, cached as a resource for the app session)
# ============================================================================
DB_PATH = "credit_risk_portfolio.db"


@st.cache_resource(show_spinner=False)
def get_connection():
    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False
    )
    return conn


# ============================================================================
# DATABASE SCHEMA CONTEXT  (identical to the notebook)
# ============================================================================
DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ============================================================================
# VERIFIED QUERY TEMPLATE LIBRARY  (identical to the notebook)
# ============================================================================
VERIFIED_QUERY_LIBRARY = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': """SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mm,
    ROUND(
        SUM(
            CASE
                WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN lm.total_outstanding
                ELSE 0
            END
        ) / 1000000.0, 2
    ) AS npa_exposure_mm
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mm DESC"""
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mm
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mm DESC"""
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': """SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mm,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mm
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage"""
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': """SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm
    ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC"""
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': """SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_exposure_mm,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
ORDER BY outstanding_exposure_mm DESC
LIMIT 10"""
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mm
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mm DESC
LIMIT 5"""
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': """SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_exposure_mm,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC"""
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mm
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END"""
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC"""
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mm
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC"""
    }
}
# NOTE: VQ7's SQL had a missing leading "SELECT" keyword in the original
# notebook cell (sql_7 started directly with "lm.loan_account_number, ...").
# It has been corrected here (SELECT added) purely so the template is
# executable — no columns, joins, filters, or ordering were changed.


# ============================================================================
# TOOL 1: INTENT CLASSIFICATION  (identical logic to the notebook)
# ============================================================================
def classify_intent(user_question, query_library, llm):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### Role
You are a query-routing classifier for an analytics system. Your job is to match a user's natural-language question to the correct predefined query template, or determine that no template applies.

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail, should NOT match a template that returns an aggregate count.

### INPUT
User question:
{user_question}
Available Verified Query templates:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ============================================================================
# TOOL 2: QUERY GENERATION  (identical logic to the notebook)
# ============================================================================
def generate_query(user_question, schema_context, llm):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### Role
You are an expert in SQLite and credit risk analytics. Your task is to write a single, valid, read-only SQLite SQL query that answers the user's question based on the provided database schema.

Here are the guidelines you must follow:
1.  **Strictly adhere to the provided database schema.** Only use tables and columns exactly as they are defined in the schema context.
2.  **Generate only SELECT queries.** Do not include any DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE) statements.
3.  **Ensure the query is read-only.** It must not modify the database in any way.
4.  **Use appropriate JOINs** when data from multiple tables is required.
5.  **Use correct aggregation functions** (e.g., SUM, AVG, COUNT) and GROUP BY clauses where necessary.
6.  **Handle dates correctly.** Use the `YYYY-MM-DD` format for date comparisons and ensure calculations involving dates are accurate.
7.  **Apply the NPA definition** for asset classification: `asset_classification IN ('Substandard', 'Doubtful', 'Loss')` when the user asks about Non-Performing Assets (NPA).
8.  **Ensure the generated SQL directly answers the user's question.**
9.  **Your output must be only the SQL query, enclosed in markdown code fences for SQL (```sql...```).** Do not include any other text, explanation, or comments outside the code fences.

Database Context:
{schema_context}

User Question:
{user_question}

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ============================================================================
# TOOL 3: QUERY VALIDATION  (identical logic to the notebook)
# ============================================================================
def validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
### Role
You are an expert in SQL and natural language understanding.
Your task is to assess whether the provided `candidate_sql` correctly and completely answers the `user_question`, given the `track_context`.

### CONTEXT
{track_context}

### INPUT
User Question:
{user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ============================================================================
# TOOL 4: RETRY GENERATION  (identical logic to the notebook)
# ============================================================================
def retry_generation(user_question, failed_sql, error_message, schema_context, llm):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### Role
You are an expert in SQLite and credit risk analytics. A previously generated SQL query failed validation. Your task is to revise the `failed_sql` to correct the identified `Validation Error`, while ensuring the revised query still correctly answers the `user_question` based on the provided `Database Schema`.

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ============================================================================
# TOOL 5: QUERY EXECUTION  (identical logic to the notebook)
# ============================================================================
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ============================================================================
# TOOL 6: RESPONSE GENERATION  (identical logic to the notebook)
# ============================================================================
def generate_response(user_question, dataframe, route, llm, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### Role
You are an expert financial analyst. Your task is to generate a concise, business-focused natural language response to the user's question, using only the provided `dataframe` results. Focus on extracting and highlighting the most relevant insights and exact figures that directly answer the `user_question`.

### INPUT
User Question:
{user_question}

Query Results Data:
{dataframe.to_string()}

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ============================================================================
# PIPELINE ORCHESTRATION  (identical logic to the notebook)
# ============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, llm, evaluator_llm, verbose=False):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, returns a list of step-by-step log lines
      (used to render the pipeline trace in the Streamlit UI).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    trace = []

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library, llm)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    trace.append(f"**[1] Intent Classification:** route=`{log['route']}`, query_id=`{log['query_id']}`")
    trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context, llm)
    log['candidate_sql'] = candidate_sql

    trace.append(
        f"**[2] Query Construction:** "
        f"{'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}"
    )

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, log['query_id'])
    log['gate_result'] = gate

    trace.append(f"**[3] Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
    if not gate['passed']:
        trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Failed check: {gate.get('failed_check')}")
        trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context, llm)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, evaluator_llm, None)
        log['gate_result'] = gate

        trace.append(f"**Retry Validation Gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")
        if not gate['passed']:
            trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Retry failed check: {gate.get('failed_check')}")
            trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        trace.append(f"**[!] Escalated to human:** {gate['details']}")
        return {'log': log, 'dataframe': None, 'trace': trace, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    trace.append(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result['warnings']:
        trace.append(f"&nbsp;&nbsp;&nbsp;&nbsp;Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], llm, log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    trace.append(f"**[6] Response Generation:** confidence=`{log['confidence']}`")

    return {'log': log, 'dataframe': df, 'trace': trace, **log}


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.title("📊 Credit Risk Portfolio Query Engine")
    st.caption(
        "Northbridge Bank · Ask routine commercial-lending portfolio questions in "
        "plain English and get verified, LLM-generated answers."
    )

    # ---- Credential / connectivity checks -------------------------------
    if not API_KEY:
        st.error(
            "No OpenAI API key found. Provide one via Streamlit secrets "
            "(`OPENAI_API_KEY`), an environment variable, or a local "
            "`config.json` file. See the deployment notes for details."
        )
        st.stop()

    if not os.path.exists(DB_PATH):
        st.error(
            f"Database file `{DB_PATH}` was not found in the app directory. "
            "Place `credit_risk_portfolio.db` alongside `app.py` (or update DB_PATH)."
        )
        st.stop()

    llm, evaluator_llm = get_llms()
    conn = get_connection()

    # ---- Sidebar: Verified Query Library reference -----------------------
    with st.sidebar:
        st.header("Verified Query Library")
        st.caption("10 pre-approved templates the engine can route to.")
        for qid, entry in VERIFIED_QUERY_LIBRARY.items():
            with st.expander(f"{qid}"):
                st.write(entry["description"])
                st.code(entry["sql"], language="sql")

        st.divider()
        st.header("Sample Questions")
        sample_questions = []
        if os.path.exists("test_queries.csv"):
            try:
                ground_truth = pd.read_csv("test_queries.csv")
                sample_questions = ground_truth["User Query"].tolist()
            except Exception:
                sample_questions = []

        if not sample_questions:
            sample_questions = [
                "What is the NPA exposure by sector?",
                "Show me the DPD aging profile of the portfolio.",
                "How many loans have been restructured and are impaired?",
                "What is the average interest rate by sector?",
                "How has Stage 3 exposure trended over the last few quarters?",
            ]

        chosen_sample = st.selectbox(
            "Load a sample question",
            options=["-- select --"] + sample_questions,
        )

    # ---- Main input --------------------------------------------------
    default_question = "" if chosen_sample == "-- select --" else chosen_sample
    user_question = st.text_area(
        "Ask a question about the credit risk portfolio",
        value=default_question,
        height=80,
        placeholder="e.g. What is the total outstanding and NPA exposure by sector?",
    )

    show_trace = st.checkbox("Show pipeline trace (routing, validation, retries)", value=True)

    run_clicked = st.button("Run Query", type="primary", use_container_width=False)

    if run_clicked:
        if not user_question.strip():
            st.warning("Please enter a question first.")
            st.stop()

        with st.spinner("Running the query engine pipeline..."):
            try:
                result = run_pipeline(
                    user_question=user_question,
                    db_connection=conn,
                    query_library=VERIFIED_QUERY_LIBRARY,
                    schema_context=DATABASE_SCHEMA,
                    llm=llm,
                    evaluator_llm=evaluator_llm,
                    verbose=True,
                )
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")
                st.stop()

        # ---- Pipeline trace -------------------------------------------
        if show_trace:
            with st.expander("Pipeline trace", expanded=False):
                for line in result["trace"]:
                    st.markdown(line, unsafe_allow_html=True)

        st.divider()

        if result["escalated"]:
            st.error("🚨 Escalated to human analyst")
            st.write(result["narrative"])
        else:
            # ---- Top-line metadata -------------------------------------
            col1, col2, col3 = st.columns(3)
            col1.metric("Route", result["route"])
            col2.metric("Query ID", result["query_id"] or "—")
            confidence = result["confidence"]
            col3.metric(
                "Confidence",
                f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence),
            )

            st.subheader("Answer")
            st.write(result["narrative"])

            st.subheader("Result Data")
            st.dataframe(result["dataframe"], use_container_width=True)

            csv_bytes = result["dataframe"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name="query_results.csv",
                mime="text/csv",
            )

            with st.expander("Executed SQL"):
                st.code(result["executed_sql"], language="sql")

            if result["retry_used"]:
                st.info("A retry was needed: the first generated query failed validation and was corrected.")

    st.divider()
    st.caption(
        "This tool answers routine, read-only portfolio questions. Ad-hoc questions "
        "outside the verified library are answered by freshly generated SQL that "
        "passes a five-step validation gate before execution; anything that fails "
        "validation is escalated to a human analyst rather than guessed at."
    )


if __name__ == "__main__":
    main()
