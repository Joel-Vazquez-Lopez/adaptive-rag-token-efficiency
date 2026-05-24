find . -name "*.py" | while read f; do
    echo ""
    echo "====================="
    echo "$f"
    echo "====================="
    grep -E "^(from|import|class|def)[[:space:]]" "$f"
done > project_map.txt