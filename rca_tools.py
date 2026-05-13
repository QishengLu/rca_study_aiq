import json
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional
from langchain_core.tools import tool

TOKEN_LIMIT = 5000

def _import_duckdb():
    """Import duckdb with helpful error message."""
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required. Install it with: pip install duckdb"
        )

def _serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj

def _estimate_token_count(text: str) -> int:
    """Estimate token count using character-based approximation.
    
    Approximate for Chinese/English mixed text.
    Average: 3 characters per token.
    """
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token

def _enforce_token_limit(payload: str, context: str) -> str:
    """Ensure payload stays within the token budget before returning"""
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload

    current_size = len(json.loads(payload)) if payload.startswith("[") else None
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))  # 80% safety margin

    suggestion_parts = [
        "The query result is too large. Please adjust your query:",
        "  • Reduce the LIMIT value" + (f" (try LIMIT {suggested_limit})" if suggested_limit else ""),
        "  • Filter rows with WHERE clauses to reduce result size",
        "  • Select only necessary columns instead of SELECT *",
        "  • Use aggregation (COUNT, SUM, AVG) instead of retrieving raw rows",
    ]

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "\n".join(suggestion_parts),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)

ALLOWED_STEMS = {
    "normal_logs", "abnormal_logs",
    "normal_traces", "abnormal_traces",
    "normal_metrics", "abnormal_metrics",
    "normal_metrics_histogram", "abnormal_metrics_histogram",
    "normal_metrics_sum", "abnormal_metrics_sum",
}

def _sanitize_column_name(name: str) -> str:
    """Replace dots in column names with underscores to avoid DuckDB dot-notation ambiguity."""
    return name.replace(".", "_")


def _build_rename_select(parquet_path: str) -> str:
    """Build a SELECT clause that renames dot-containing columns for a parquet file.

    Returns 'SELECT col1, "attr.x" AS attr_x, ...' or 'SELECT *' if no renames needed.
    """
    duckdb = _import_duckdb()
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0")
        columns = [desc[0] for desc in result.description]
    finally:
        conn.close()

    needs_rename = any("." in col for col in columns)
    if not needs_rename:
        return "*"

    parts = []
    for col in columns:
        if "." in col:
            parts.append(f'"{col}" AS {_sanitize_column_name(col)}')
        else:
            parts.append(col)
    return ", ".join(parts)


def _validate_parquet_files(parquet_files: Union[str, List[str]]) -> List[str]:
    """Validate parquet files exist and return as list."""
    if isinstance(parquet_files, str):
        # Some models (e.g. Qwen) serialize list args as JSON strings:
        #   '["file1.parquet", "file2.parquet"]' or '"file.parquet"'
        stripped = parquet_files.strip()
        if stripped.startswith("["):
            try:
                parquet_files = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                parquet_files = [parquet_files]
        elif stripped.startswith('"') and stripped.endswith('"'):
            parquet_files = [stripped.strip('"')]
        else:
            parquet_files = [parquet_files]

    for file_path in parquet_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Parquet file not found: {file_path}\n"
                f"Please check the file path and ensure the file exists. "
                f"You may use 'list_tables_in_directory' to discover available parquet files."
            )
    return parquet_files

@tool
def list_tables_in_directory(directory: str) -> str:
    """
    List all parquet files in a directory with metadata.

    Args:
        directory: Directory path to search for parquet files

    Returns:
        JSON string containing list of files with metadata
    """
    duckdb = _import_duckdb()

    dir_path = Path(directory)
    if not dir_path.exists():
        return json.dumps({"error": f"Directory not found: {directory}"})

    if not dir_path.is_dir():
        return json.dumps({"error": f"Path is not a directory: {directory}"})

    files_info = []
    cwd = Path.cwd()

    # Use rglob to find parquet files recursively (only allowed stems)
    for file_path in sorted(dir_path.rglob("*.parquet")):
        if Path(file_path).stem not in ALLOWED_STEMS:
            continue
        file_path_str = str(file_path)
        file_path_obj = Path(file_path_str)
        if file_path_obj.is_absolute():
            try:
                file_path_str = str(file_path_obj.relative_to(cwd))
            except ValueError:
                file_path_str = str(file_path_obj)
        
        try:
            conn = duckdb.connect(":memory:")
            row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()
            if row_count_result is None:
                row_count = 0
            else:
                row_count = row_count_result[0]
            
            result = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
            column_count = len(result.description)
            conn.close()

            files_info.append(
                {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "row_count": row_count,
                    "column_count": column_count,
                }
            )
        except Exception as e:
            files_info.append({
                "filename": file_path.name, 
                "path": str(file_path), 
                "error": str(e)
            })

    result_json = json.dumps(files_info, ensure_ascii=False, indent=2)
    return _enforce_token_limit(result_json, "list_tables_in_directory")

def _get_schema_one(parquet_file: str) -> dict:
    """Get schema for a single parquet file, returning a dict."""
    duckdb = _import_duckdb()

    if not Path(parquet_file).exists():
        return {"error": f"Parquet file not found: {parquet_file}"}

    conn = duckdb.connect(":memory:")
    try:
        cwd = Path.cwd()
        parquet_file_obj = Path(parquet_file)
        if parquet_file_obj.is_absolute():
            try:
                parquet_file_rel = str(parquet_file_obj.relative_to(cwd))
            except ValueError:
                parquet_file_rel = str(parquet_file_obj)
        else:
            parquet_file_rel = parquet_file

        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 0")
        schema = [{"name": _sanitize_column_name(desc[0]), "type": str(desc[1])} for desc in result.description]

        row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
        if row_count_result is None:
            row_count = 0
        else:
            row_count = row_count_result[0]

        return {
            "file": parquet_file,
            "row_count": row_count,
            "columns": schema,
        }

    except Exception as e:
        return {"error": f"Failed to extract schema: {str(e)}"}
    finally:
        conn.close()


@tool
def get_schema(parquet_files: Union[str, List[str]]) -> str:
    """
    Get schema information of a parquet file, or a list of parquet files.

    Args:
        parquet_files: Path to a parquet file, or list of paths for batch lookup

    Returns:
        JSON string containing file metadata — single object if one file, list if multiple
    """
    # Normalize: some models serialize list args as JSON strings
    if isinstance(parquet_files, str):
        stripped = parquet_files.strip()
        if stripped.startswith("["):
            try:
                parquet_files = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
        elif stripped.startswith('"') and stripped.endswith('"'):
            parquet_files = stripped.strip('"')

    if isinstance(parquet_files, str):
        result_json = json.dumps(_get_schema_one(parquet_files), ensure_ascii=False, indent=2)
    else:
        result_json = json.dumps([_get_schema_one(f) for f in parquet_files], ensure_ascii=False, indent=2)

    return _enforce_token_limit(result_json, "get_schema")

@tool
def query_parquet_files(parquet_files: Union[str, List[str]], query: str, limit: int = 10) -> str:
    """
    Query parquet files using SQL syntax for data analysis and exploration.
    
    Args:
        parquet_files: Path(s) to parquet file(s)
        query: SQL query to execute
        limit: Maximum number of records to return
        
    Returns:
        JSON string of query results
    """
    duckdb = _import_duckdb()
    try:
        parquet_files = _validate_parquet_files(parquet_files)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})

    conn = duckdb.connect(":memory:")
    table_names: set = set()

    try:
        # Convert absolute paths to relative paths where possible
        cwd = Path.cwd()
        relative_parquet_files = []
        for file_path in parquet_files:
            file_path_obj = Path(file_path)
            if file_path_obj.is_absolute():
                try:
                    file_path = str(file_path_obj.relative_to(cwd))
                except ValueError:
                    # File is not under cwd (e.g., temp files), use absolute path
                    file_path = str(file_path_obj)
            relative_parquet_files.append(file_path)
        parquet_files = relative_parquet_files

        # Register parquet files as views using filename as table name
        for file_path in parquet_files:
            base_name = Path(file_path).stem
            table_name = base_name
            counter = 1
            while table_name in table_names:
                table_name = f"{base_name}_{counter}"
                counter += 1
            table_names.add(table_name)
            select_clause = _build_rename_select(file_path)
            conn.execute(f"CREATE VIEW {table_name} AS SELECT {select_clause} FROM read_parquet('{file_path}')")

        # Execute query
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]

        # Convert to list of dictionaries and serialize datetime
        rows = [dict(zip(columns, row)) for row in result]
        serialized_rows = _serialize_datetime(rows)

        # Apply limit if specified
        if len(serialized_rows) > limit:
            serialized_rows = serialized_rows[:limit]

        result_json = json.dumps(serialized_rows, ensure_ascii=False, indent=2)
        return _enforce_token_limit(result_json, "query_parquet_files")

    except Exception as e:
        error_msg = str(e)
        # Provide contextual error messages
        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return json.dumps({
                "error": f"SQL syntax error in query: {error_msg}",
                "query": query,
                "available_tables": list(table_names)
            })
        elif "catalog" in error_msg.lower() or "table" in error_msg.lower():
            return json.dumps({
                "error": f"Table reference error: {error_msg}",
                "query": query,
                "available_tables": list(table_names)
            })
        else:
            return json.dumps({
                "error": f"Query execution failed: {error_msg}",
                "query": query,
                "available_tables": list(table_names)
            })
    finally:
        conn.close()
