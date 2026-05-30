#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <cctype>

// =============================================================
//  Vocabulary
//
//  Builds a word->index map from training sentences.
//  Converts sentences to fixed-length token sequences.
//
//  Special tokens:
//    <PAD> = 0   padding for short sentences
//    <UNK> = 1   unknown words at inference time
//
//  Usage:
//    Vocabulary vocab;
//    vocab.build(sentences);               // learns word->index
//    auto seq = vocab.encode("turn on the lights", 10);
//    // seq = [0.12, 0.04, 0.36, 0.0, ...]  (normalized token IDs)
// =============================================================

class Vocabulary
{
public:
    Vocabulary()
    {
      // reserve special tokens
      wordToIdx_["<PAD>"] = 0;
      wordToIdx_["<UNK>"] = 1;
      idxToWord_.push_back("<PAD>");
      idxToWord_.push_back("<UNK>");
    }

    // ── build from a list of raw sentences ──────────────────
    void build(const std::vector<std::string>& sentences)
    {
      for (const auto& s : sentences)
      {
        for (const auto& tok : tokenise(s))
        {
          if (wordToIdx_.find(tok) == wordToIdx_.end())
          {
            int idx = (int)idxToWord_.size();
            wordToIdx_[tok] = idx;
            idxToWord_.push_back(tok);
          }
        }
      }
    }

    // ── encode sentence → fixed-length float vector ─────────
    // Each position = tokenID / vocabSize  (normalised to [0,1])
    // Short sentences are zero-padded. Long ones are truncated.
    std::vector<double> encode(const std::string& sentence, int seqLen) const
    {
      auto tokens = tokenise(sentence);
      std::vector<double> out(seqLen * size(), 0.0);  // One-hot flattened

      for (int i = 0; i < seqLen && i < (int)tokens.size(); i++)
      {
        auto it = wordToIdx_.find(tokens[i]);
        int id = (it != wordToIdx_.end()) ? it->second : 1;  // <UNK>

        out[i * size() + id] = 1.0;
      }
      return out;
    }

    int         size()              const { return (int)idxToWord_.size(); }
    std::string word(int idx)       const { return idxToWord_[idx]; }
    int         index(const std::string& w) const
    {
      auto it = wordToIdx_.find(w);
      return (it != wordToIdx_.end()) ? it->second : 1;
    }

    // ── simple whitespace + lowercase tokeniser ──────────────
    static std::vector<std::string> tokenise(const std::string& s)
    {
      std::vector<std::string> tokens;
      std::string lower;
      lower.reserve(s.size());

      // lowercase and strip punctuation
      for (char c : s)
      {
        if (std::isalpha(c)) lower += std::tolower(c);
        else if (std::isspace(c)) lower += ' ';
      }

      std::istringstream ss(lower);
      std::string tok;
      while (ss >> tok)
      {
        tokens.push_back(tok);
      }
      return tokens;
    }

    void printVocab() const
    {
      std::cout << "Vocabulary (" << size() << " tokens):\n";
      for (int i = 0; i < (int)idxToWord_.size(); i++)
      {
        std::cout << "  [" << i << "] " << idxToWord_[i] << "\n";
      }
    }

private:
    std::unordered_map<std::string, int> wordToIdx_;
    std::vector<std::string>             idxToWord_;
};
