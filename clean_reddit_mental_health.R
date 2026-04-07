# =============================================================================
# clean_reddit_mental_health.R
# =============================================================================
# NLP Preprocessing Pipeline for Reddit Mental Health Dataset
# -----------------------------------------------------------------------------
# This script cleans and prepares a Reddit mental health CSV dataset
# for downstream NLP feature extraction. It performs three main stages:
#
#   1. Placeholder & Noise Removal  — strips values that carry no linguistic signal
#   2. Feature Engineering          — builds a structured `text` column
#   3. Row-Level Deduplication      — removes reposts and bot-duplicated content
#
# Input  : combine.csv          (raw Reddit dataset)
# Output : combine_cleaned.csv  (cleaned, deduplicated dataset with `text` column)
#
# Dependencies: tidyverse (dplyr, readr, stringr)
# Install via: install.packages("tidyverse")
# =============================================================================

library(dplyr)
library(readr)
library(stringr)


# ── 0. LOAD ───────────────────────────────────────────────────────────────────
# Read the raw dataset using readr for fast parsing and clean column types.

df <- read_csv("combine.csv", show_col_types = FALSE)

cat("[LOAD] Rows loaded :", nrow(df), "\n")
cat("[LOAD] Columns     :", paste(colnames(df), collapse = ", "), "\n\n")


# ── 1. PLACEHOLDER & NOISE REMOVAL ────────────────────────────────────────────
#
# Reddit posts often contain values that look like content but carry zero
# linguistic meaning. Keeping them would introduce noise into embeddings or
# bag-of-words features. We replace them with empty strings so downstream
# concatenation still works — we do NOT drop the row, because the title or
# subreddit alone may still be informative.
#
# selftext noise values:
#   • NA         — post had no body (link-only or deleted before crawl)
#   • [removed]  — Reddit's automated or moderator removal marker
#   • [deleted]  — user deleted their own post body
#   • "."        — common placeholder used to satisfy Reddit's minimum-body rule
#
# title noise values:
#   • "."        — same placeholder pattern occasionally appears in titles

SELFTEXT_NOISE <- c("[removed]", "[deleted]", ".")

df <- df %>%
  mutate(
    # Replace NA and known noise tokens in selftext with empty string
    selftext = if_else(
      is.na(selftext) | selftext %in% SELFTEXT_NOISE,
      "",
      selftext
    ),

    # Replace NA and dot placeholder in title with empty string
    title = if_else(
      is.na(title) | title == ".",
      "",
      title
    )
  )

cat("[CLEAN] Noise replacement complete.\n")
cat("        selftext : NA, [removed], [deleted], and '.' -> \"\"\n")
cat("        title    : '.' -> \"\"\n\n")


# ── 2. FEATURE ENGINEERING — `text` COLUMN ────────────────────────────────────
#
# We concatenate three fields into a single structured string.
# The bracketed tags serve two purposes:
#   a) They preserve the semantic role of each segment for models that
#      benefit from token-level structure (e.g. BERT with special tokens).
#   b) They make the column human-readable for manual inspection.
#
# Format:
#   "[SUBREDDIT: <name>] [TITLE: <title_text>] [BODY: <selftext>]"
#
# If selftext or title was empty after cleaning, the segment still appears
# as "[BODY: ]" — consistent formatting avoids parsing edge cases later.

df <- df %>%
  mutate(
    text = str_c(
      "[SUBREDDIT: ", subreddit, "] ",
      "[TITLE: ",    title,     "] ",
      "[BODY: ",     selftext,  "]"
    )
  )

cat("[FEATURE] `text` column created. Sample:\n")
cat(" ", str_trunc(df$text[1], width = 180), "...\n\n")


# ── 3. ROW-LEVEL DEDUPLICATION ────────────────────────────────────────────────
#
# Reddit is prone to reposts and bot activity that submits the same content
# multiple times, sometimes across subreddits. Training a model on duplicate
# text inflates the effective weight of those examples and can cause
# overfitting to recurring patterns.
#
# We deduplicate on the full `text` column (exact string match) because it
# encodes subreddit + title + body together — a post resubmitted to a
# different subreddit will NOT be collapsed, only true verbatim duplicates.
#
# distinct() with .keep_all = TRUE preserves all columns and retains
# the first occurrence of each unique `text` value.

rows_before <- nrow(df)

df <- df %>%
  distinct(text, .keep_all = TRUE)

rows_after <- nrow(df)

cat("[DEDUP] Rows before :", format(rows_before, big.mark = ","), "\n")
cat("[DEDUP] Rows after  :", format(rows_after,  big.mark = ","), "\n")
cat("[DEDUP] Removed     :", format(rows_before - rows_after, big.mark = ","), "duplicate rows\n\n")


# ── 4. SAVE ───────────────────────────────────────────────────────────────────
output_path <- "combine_cleaned.csv"
write_csv(df, output_path)

cat("[SAVE] Cleaned dataset written to '", output_path, "'\n", sep = "")
cat("       Final shape :", nrow(df), "rows x", ncol(df), "columns\n")
