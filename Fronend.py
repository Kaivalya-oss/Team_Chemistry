from flask import Flask, request, jsonify
import pandas as pd
from step11 import simulate_player_swap   # your function above

app = Flask(__name__)

players = pd.read_csv("players_for_dashboard.csv")
teams = pd.read_csv("teams_for_dashboard.csv")

@app.route('/')
def index():
    return """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <h1>Team Chemistry Dashboard</h1>
    <label>Team:</label>
    <select id="team"></select><br><br>
    
    <label>Remove Player:</label>
    <select id="remove_player"></select><br><br>

    <label>Add Player:</label>
    <select id="add_player"></select><br><br>
    
    <button onclick="simulate()">Simulate Swap</button>
    <div id="result"></div>
    <div style="width: 500px; height: 500px;">
        <canvas id="radarChart"></canvas>
    </div>

    <script>
    fetch('/teams').then(r=>r.json()).then(data => {
      const sel = document.getElementById('team');
      data.forEach(t => {
        let opt = document.createElement('option');
        opt.value = t; opt.text = t;
        sel.add(opt);
      });
    });

    document.getElementById('team').onchange = function() {
      const team = this.value;
      
      // 1. Get current team members (to remove)
      fetch(`/team_members/${encodeURIComponent(team)}`)
        .then(r=>r.json())
        .then(data => {
          const sel = document.getElementById('remove_player');
          sel.innerHTML = '<option value="">-- Select to Remove --</option>';
          data.forEach(p => {
            let opt = document.createElement('option');
            opt.value = p; opt.text = p;
            sel.add(opt);
          });
        });

      // 2. Get candidates (to add)
      fetch(`/players/${encodeURIComponent(team)}`)
        .then(r=>r.json())
        .then(data => {
          const sel = document.getElementById('add_player');
          sel.innerHTML = '<option value="">-- Select to Add --</option>';
          data.forEach(p => {
            let opt = document.createElement('option');
            opt.value = p; opt.text = p;
            sel.add(opt);
          });
        });
    };

    function simulate() {
      const team = document.getElementById('team').value;
      const remove_player = document.getElementById('remove_player').value;
      const add_player = document.getElementById('add_player').value;
      
      if (!team || !remove_player || !add_player) return alert("Select team, player to remove, and player to add");
      
      fetch('/simulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({team, remove_player, add_player})
      })
      .then(r=>r.json())
      .then(res => {
        if (res.error) {
            document.getElementById('result').innerHTML = `<h3 style="color:red">Error: ${res.error}</h3>`;
        } else {
            document.getElementById('result').innerHTML = `
              <h3>${res.team}</h3>
              <p>Replaced: <b>${res.removed}</b> &rarr; <b>${res.added}</b></p>
              <p>Before: <b>${res.before_chemistry}</b></p>
              <p>After: <b>${res.after_chemistry}</b></p>
              <p style="color:${res.change > 0 ? 'green' : 'red'}">
                Change: ${res.change > 0 ? '+' : ''}${res.change}
              </p>
            `;
            
            // Draw Radar Chart
            const ctx = document.getElementById('radarChart').getContext('2d');
            
            // Destroy existing chart if it exists
            if (window.myRadarChart) {
                window.myRadarChart.destroy();
            }
            
            window.myRadarChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: res.metrics.labels,
                    datasets: [{
                        label: 'Before',
                        data: res.metrics.before,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    }, {
                        label: 'After',
                        data: res.metrics.after,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    }]
                },
                options: {
                    scale: {
                        ticks: { beginAtZero: true, max: 1.0 }
                    }
                }
            });
        }
      });
    }
    </script>
    """

@app.route('/teams')
def get_teams():
    return jsonify(teams['Club'].tolist())

@app.route('/team_members/<team>')
def get_team_members(team):
    # Get players currently in the team
    members = players[players['Club'] == team]['Name'].tolist()
    return jsonify(members)

@app.route('/players/<team>')
def get_players(team):
    # Get players NOT in the selected team
    # Optimize: Filter potential players first, then select top 300 by Reputation
    candidates = players[players['Club'] != team]
    
    # Sort by Reputation (descending), then Name
    candidates = candidates.sort_values(by=['International Reputation', 'Name'], ascending=[False, True])
    
    # Take top 300 unique names
    other_players = candidates['Name'].unique()[:300].tolist()
    
    return jsonify(other_players)

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    result = simulate_player_swap(data['team'], data['remove_player'], data['add_player'])
    return jsonify(result)

if __name__ == '__main__':
    # Data updated, restarting app
    app.run(debug=True)