
const completedHashes = require('./completed_hashes.json');
const express = require('express');
const path = require('path');
const PORT = 3333;
const fs = require('fs');

const rawJson = fs.readFileSync('../alpaca_data_cleaned.json', 'utf-8');
const data = JSON.parse(rawJson);
escapeUnicodeInValues(data);

function escapeUnicode(str) {
    return str.replace(/[\u007F-\uFFFF]/g, (char) => {
        const unicode = '\\u' + ('000' + char.charCodeAt(0).toString(16)).slice(-4);
        return unicode
    });
}

function escapeUnicodeInValues(obj) {
    for (const key in obj) {
      if (typeof obj[key] === 'string') {
        obj[key] = escapeUnicode(obj[key]);
      } else if (typeof obj[key] === 'object' && obj[key] !== null) {
        escapeUnicodeInValues(obj[key]);
      }
    }
  }

function escapeUnicodeReplacer(key, value) {
    if (typeof value === 'string') {
        return escapeUnicode(value);
    }
    return value;
}

function getRandomEntry() {
    const randomIndex = Math.floor(Math.random() * data.length);
    const randomEntry = data[randomIndex];
    const { instruction, input, output } = randomEntry;
    const hash = `${instruction}::::${input};;;;;${output}`
    if (completedHashes[hash]) {
        return getRandomEntry();
    }
    return [randomEntry, randomIndex];
}

function markAsCompleted(entry) {
    const { instruction, input, output } = entry;
    const hash = `${instruction}::::${input};;;;;${output}`
    completedHashes[hash] = true;
    fs.writeFileSync('./completed_hashes.json', JSON.stringify(completedHashes, null, 4));
}

function updateEntry(index, entry) {
    data[index] = entry;
    const string = JSON.stringify(data, escapeUnicodeReplacer, 4);
    fs.writeFileSync('../alpaca_data_cleaned.json', string.replaceAll("\\\\u", "\\u"), "utf-8");
    markAsCompleted(entry);
}

function deleteEntry(index) {
    data.splice(index, 1);
    const string = JSON.stringify(data, escapeUnicodeReplacer, 4);
    fs.writeFileSync('../alpaca_data_cleaned.json', string.replaceAll("\\\\u", "\\u"), "utf-8");
}

const app = express();
app.use(express.json());
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'page.html'));
});
app.get("/api/entry", (req, res) => {
    const [entry, index] = getRandomEntry();
    res.send(JSON.stringify({ entry, index }, null, 2));
});
app.post("/api/entry", (req, res) => {
    const { index, entry } = req.body;
    updateEntry(index, entry);
    res.json({ success: true });
});
app.delete("/api/entry", (req, res) => {
    const { index } = req.body;
    deleteEntry(index);
    res.json({ success: true });
});
app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
    console.log('http://localhost:3333')
})