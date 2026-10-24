// Test-only: scan song folders with scan-chart (what Clone Hero's own parser
// does) and print one JSON line per folder. Not bundled with BandSmith.
//
//     node check.mjs <song folder> [<song folder> ...]
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { scanChartFolder } from 'scan-chart'

for (const dir of process.argv.slice(2)) {
  const files = readdirSync(dir)
    .filter(f => statSync(join(dir, f)).isFile())
    .map(f => ({ fileName: f, data: new Uint8Array(readFileSync(join(dir, f))) }))
  const s = scanChartFolder(files, { includeMd5: false })
  const n = s.notesData
  console.log(JSON.stringify({
    folder: dir,
    playable: s.playable,
    name: s.name,
    song_length: s.song_length,
    folderIssues: s.folderIssues,
    metadataIssues: s.metadataIssues,
    chartIssues: n ? n.chartIssues : [],
    instruments: n ? n.instruments : [],
    drumType: n ? n.drumType : null,
    hasVocals: n ? n.hasVocals : false,
    hasLyrics: n ? n.hasLyrics : false,
    noteCounts: n ? n.noteCounts : [],
    tempoMarkerCount: n ? n.tempoMarkerCount : 0,
    effectiveLength: n ? n.effectiveLength : 0,
  }))
}
