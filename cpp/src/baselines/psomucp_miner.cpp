#include "baselines/psomucp_miner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace colocation {

std::map<char, double> PsoMucpConfig::parseUtility(std::string text) {
    std::string compact;
    for (char c : text)
        if (c != '{' && c != '}' && c != '\'' && c != '"' && c != ' ') compact += c;
    std::map<char, double> out;
    for (const std::string& item : CsvTable::splitLine(compact, ',')) {
        const std::size_t colon = item.find(':');
        if (colon != std::string::npos && colon > 0) out[item[0]] = std::stod(item.substr(colon + 1));
    }
    return out;
}

PsoMucpMiner::PsoMucpMiner(std::vector<PsoInstance> instances, PsoMucpConfig config)
    : instances_(std::move(instances)), config_(std::move(config)) {
    if (instances_.empty()) throw std::invalid_argument("no instances");
}

std::vector<PsoInstance> PsoMucpMiner::loadInstances(const CsvTable& table) {
    std::vector<PsoInstance> out;
    for (const auto& r : table.rows()) out.push_back({r[0][0], r[0] + r[1], std::stod(r[2]), std::stod(r[3])});
    return out;
}

bool PsoMucpMiner::contains(const std::vector<char>& v, char c) { return std::find(v.begin(), v.end(), c) != v.end(); }

double PsoMucpMiner::utilityOf(char feature) const {
    auto it = config_.utility.find(feature);
    if (it == config_.utility.end()) throw std::invalid_argument(std::string("no utility for feature ") + feature);
    return it->second;
}

// Compute_utility: sum over the features of the utility dict of count x utility (all signs)
double PsoMucpMiner::totalUtility() const {
    double total = 0;
    for (const auto& kv : config_.utility) {
        long long count = 0;
        for (const auto& o : instances_)
            if (o.feature == kv.first) ++count;
        total += count * kv.second;
    }
    return total;
}

// grid_method: cells of size D anchored at the minimum coordinates (0 becomes 1), own cell then the 8 around it
void PsoMucpMiner::buildNeighborhoods() {
    const int n = static_cast<int>(instances_.size());
    double xmin = instances_[0].x, ymin = instances_[0].y;
    for (const auto& o : instances_) {
        xmin = std::min(xmin, o.x);
        ymin = std::min(ymin, o.y);
    }
    const double D = config_.distance;
    std::map<std::pair<long long, long long>, int> cellIndex;
    std::vector<std::pair<long long, long long>> cells;   // insertion order of the Python dict
    std::vector<std::vector<int>> cellMembers;
    for (int i = 0; i < n; ++i) {
        long long cx = static_cast<long long>(std::ceil((instances_[i].x - xmin) / D));
        long long cy = static_cast<long long>(std::ceil((instances_[i].y - ymin) / D));
        if (cx == 0) cx = 1;
        if (cy == 0) cy = 1;
        auto it = cellIndex.find({cx, cy});
        if (it == cellIndex.end()) {
            cellIndex[{cx, cy}] = static_cast<int>(cells.size());
            cells.push_back({cx, cy});
            cellMembers.push_back({i});
        } else {
            cellMembers[it->second].push_back(i);
        }
    }
    neighbors_.assign(n, {});
    const double limit = D * D;
    auto close = [&](int a, int b) {
        const double dx = instances_[a].x - instances_[b].x, dy = instances_[a].y - instances_[b].y;
        return dx * dx + dy * dy <= limit;
    };
    auto addFrom = [&](int e, const std::vector<int>& members) {
        for (int x : members)
            if (instances_[x].feature != instances_[e].feature && close(x, e)) neighbors_[e].push_back(x);
    };
    for (std::size_t c = 0; c < cells.size(); ++c)
        for (int e : cellMembers[c]) {
            neighborhoodOrder_.push_back(e);
            addFrom(e, cellMembers[c]);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    auto it = cellIndex.find({cells[c].first + dx, cells[c].second + dy});
                    if (it != cellIndex.end()) addFrom(e, cellMembers[it->second]);
                }
        }
}

// Prune_one: feature upper bound Up(f) = positive utility of the instances of f and their neighbours; features with
// Up / ALL <= minutil removed; positive features by increasing Up, then the others
void PsoMucpMiner::pruneFeatures() {
    std::map<char, std::unordered_set<int>> reach;
    std::vector<char> featureOrder;
    for (int key : neighborhoodOrder_) {
        const char f = instances_[key].feature;
        if (!reach.count(f)) {
            reach[f];
            featureOrder.push_back(f);
            (utilityOf(f) > 0 ? positive_ : negative_).push_back(f);
        }
        auto& set = reach[f];
        for (int x : neighbors_[key]) set.insert(x);
        set.insert(key);
    }
    std::map<char, double> up;
    for (char f : featureOrder) {
        double s = 0;
        for (int x : reach[f])
            if (utilityOf(instances_[x].feature) > 0) s += utilityOf(instances_[x].feature);
        up[f] = s;
    }
    std::vector<char> keptPositive, keptNegative;
    for (char f : featureOrder) {
        if (up[f] / normalizer_ <= config_.minUtility) deleted_.push_back(f);
        else (contains(positive_, f) ? keptPositive : keptNegative).push_back(f);
    }
    auto byBound = [&](char a, char b) { return up[a] < up[b]; };
    std::stable_sort(keptPositive.begin(), keptPositive.end(), byBound);
    std::stable_sort(keptNegative.begin(), keptNegative.end(), byBound);
    sortList_ = keptPositive;
    sortList_.insert(sortList_.end(), keptNegative.begin(), keptNegative.end());
    // delete_nis
    const int n = static_cast<int>(instances_.size());
    keptNeighbors_.assign(n, {});
    keptNeighborSet_.assign(n, {});
    for (int key : neighborhoodOrder_) {
        if (contains(deleted_, instances_[key].feature)) continue;
        keptOrder_.push_back(key);
        for (int x : neighbors_[key])
            if (!contains(deleted_, instances_[x].feature)) keptNeighbors_[key].push_back(x);
        keptNeighborSet_[key] = std::unordered_set<int>(keptNeighbors_[key].begin(), keptNeighbors_[key].end());
    }
}

// con_hash_table for one clique of `key`'s neighbourhood: key = sorted features of key + clique
void PsoMucpMiner::addClique(int key, const std::vector<int>& clique) {
    ++cliqueCount_;
    std::string hashKey(1, instances_[key].feature);
    for (int e : clique) hashKey += instances_[e].feature;
    std::sort(hashKey.begin(), hashKey.end());
    auto it = hashIndex_.find(hashKey);
    int id;
    if (it == hashIndex_.end()) {
        id = static_cast<int>(table_.size());
        hashIndex_[hashKey] = id;
        hashKeys_.push_back(hashKey);
        table_.emplace_back();
    } else {
        id = it->second;
    }
    HashEntry& entry = table_[id];
    auto put = [&](int o) {
        const char f = instances_[o].feature;
        if (!entry.instances.count(f)) entry.features.push_back(f);
        if (entry.seen[f].insert(o).second) entry.instances[f].push_back(o);
    };
    put(key);
    for (int e : clique) put(e);
}

// Enum_Cliques: for every key and every neighbour `current` (in list order), the cliques of current with the later
// neighbours of key, by the include/exclude depth-first search of the Python code
void PsoMucpMiner::enumerateCliques() {
    struct Frame {
        std::vector<int> clique, candidates;
    };
    for (int key : keptOrder_) {
        const std::vector<int>& list = keptNeighbors_[key];
        for (std::size_t head = 0; head < list.size(); ++head) {
            const int current = list[head];
            std::vector<int> candidates;
            for (std::size_t t = head + 1; t < list.size(); ++t)
                if (keptNeighborSet_[current].count(list[t])) candidates.push_back(list[t]);
            std::vector<Frame> stack;
            stack.push_back({{current}, candidates});
            while (!stack.empty()) {
                Frame frame = std::move(stack.back());
                stack.pop_back();
                if (frame.candidates.empty()) {
                    addClique(key, frame.clique);
                    continue;
                }
                if (frame.candidates.size() == 1) {
                    std::vector<int> c = frame.clique;
                    c.push_back(frame.candidates[0]);
                    addClique(key, c);
                    continue;
                }
                const int first = frame.candidates[0];
                std::vector<int> with = frame.clique;
                with.push_back(first);
                std::vector<int> remaining(frame.candidates.begin() + 1, frame.candidates.end());
                std::vector<int> withCandidates;
                for (int x : remaining)
                    if (keptNeighborSet_[first].count(x)) withCandidates.push_back(x);
                stack.push_back({with, withCandidates});
                stack.push_back({frame.clique, remaining});
            }
        }
    }
}

bool PsoMucpMiner::laterInOrder(char feature, const std::string& pattern) const {
    int lastPos = -1, featurePos = -1;
    for (std::size_t i = 0; i < sortList_.size(); ++i) {
        if (sortList_[i] == pattern.back()) lastPos = static_cast<int>(i);
        if (sortList_[i] == feature) featurePos = static_cast<int>(i);
    }
    return featurePos > lastPos;
}

bool PsoMucpMiner::keyContains(const std::string& key, const std::string& pattern) const {
    for (char c : pattern)
        if (key.find(c) == std::string::npos) return false;
    return true;
}

// one level of Find_High_utility_pattern: utility (participating instances of the pattern's features in all keys
// that contain the pattern) and PUUR (instances of the pattern's positive features and of the positive features after
// it, in all such keys except a key equal to the candidate string)
std::map<std::string, PsoMucpMiner::Candidate> PsoMucpMiner::scoreCandidates(const std::vector<std::string>& candidates) const {
    std::map<std::string, Candidate> scores;
    std::map<std::string, std::map<char, std::unordered_set<int>>> counted, countedUp;
    for (const auto& c : candidates) scores[c];
    for (std::size_t h = 0; h < table_.size(); ++h) {
        const std::string& key = hashKeys_[h];
        const HashEntry& entry = table_[h];
        for (const std::string& cand : candidates) {
            if (!keyContains(key, cand)) continue;
            std::map<char, std::vector<int>> fresh, freshUp;
            auto& seen = counted[cand];
            auto& seenUp = countedUp[cand];
            for (char f : entry.features) {
                const bool inPattern = cand.find(f) != std::string::npos;
                const bool isNegative = contains(negative_, f);
                if (inPattern) {
                    auto& list = fresh[f];
                    auto& s = seen[f];
                    for (int o : entry.instances.at(f))
                        if (!s.count(o)) list.push_back(o);
                }
                const bool later = laterInOrder(f, cand);
                if (!freshUp.count(f) && (inPattern || later) && !isNegative) {
                    freshUp[f];
                    seenUp[f];
                }
                if ((!inPattern && !later) || isNegative) continue;
                auto& s = seenUp[f];
                for (int o : entry.instances.at(f))
                    if (key != cand && !s.count(o)) freshUp[f].push_back(o);
            }
            for (auto& kv : fresh)
                for (int o : kv.second) seen[kv.first].insert(o);
            for (auto& kv : freshUp)
                for (int o : kv.second) seenUp[kv.first].insert(o);
            Candidate& sc = scores[cand];
            for (auto& kv : fresh) sc.utility += static_cast<double>(kv.second.size()) * utilityOf(kv.first);
            for (auto& kv : freshUp) sc.upperBound += static_cast<double>(kv.second.size()) * utilityOf(kv.first);
        }
    }
    return scores;
}

// gener_k: extensions by later features of the sort list; PUUR / ALL < minutil -> no-hope prefix (Strategy 2);
// lambda / ALL <= minutil -> no negative extension (Strategy 4); the extension must be contained in some hash key
std::vector<std::string> PsoMucpMiner::generateCandidates(const std::vector<std::string>& candidates,
                                                          const std::map<std::string, Candidate>& scores,
                                                          std::vector<std::string>& noHope) const {
    std::vector<std::string> next;
    std::unordered_set<std::string> added;
    for (const std::string& base : candidates) {
        const Candidate& sc = scores.at(base);
        if (sc.upperBound / normalizer_ < config_.minUtility) {
            noHope.push_back(base);
            continue;
        }
        int lastPos = 0;
        for (std::size_t j = 0; j < sortList_.size(); ++j)
            if (base.back() == sortList_[j]) {
                lastPos = static_cast<int>(j);
                break;
            }
        for (std::size_t i = 0; i < sortList_.size(); ++i) {
            if (sc.utility / normalizer_ <= config_.minUtility && contains(negative_, sortList_[i])) continue;
            if (lastPos >= static_cast<int>(i)) continue;
            if (base.find(sortList_[i]) != std::string::npos) continue;
            const std::string ext = base + sortList_[i];
            bool hopeless = false;
            for (const auto& prefix : noHope)
                if (ext.size() >= prefix.size() && ext.compare(0, prefix.size(), prefix) == 0) {
                    hopeless = true;
                    break;
                }
            if (hopeless) continue;
            bool exists = false;
            for (const auto& key : hashKeys_)
                if (keyContains(key, ext)) {
                    exists = true;
                    break;
                }
            if (exists && added.insert(ext).second) next.push_back(ext);
        }
    }
    return next;
}

PsoMucpResult PsoMucpMiner::mine() {
    normalizer_ = config_.normalizer ? *config_.normalizer : totalUtility();
    buildNeighborhoods();
    pruneFeatures();
    enumerateCliques();
    PsoMucpResult result;
    result.normalizer = normalizer_;
    std::vector<std::string> candidates, noHope, high;
    for (char c : sortList_)
        if (!contains(negative_, c)) candidates.emplace_back(1, c);
    while (!candidates.empty()) {
        result.evaluated += static_cast<long long>(candidates.size());
        const auto scores = scoreCandidates(candidates);
        for (const auto& c : candidates)
            if (scores.at(c).utility / normalizer_ >= config_.minUtility) high.push_back(c);
        candidates = generateCandidates(candidates, scores, noHope);
        result.generatedCandidates += static_cast<long long>(candidates.size());
    }
    for (const auto& p : high)
        if (p.size() > 1) {
            std::string s = p;
            std::sort(s.begin(), s.end());
            result.patterns.push_back(s);
        }
    std::sort(result.patterns.begin(), result.patterns.end());
    result.patterns.erase(std::unique(result.patterns.begin(), result.patterns.end()), result.patterns.end());
    result.cliques = cliqueCount_;
    result.hashKeys = static_cast<long long>(table_.size());
    return result;
}

}  // namespace colocation
