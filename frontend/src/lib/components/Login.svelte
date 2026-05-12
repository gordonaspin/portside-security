<script>
  let username = "";
  let password = "";
  let error = "";

  async function login() {
    error = "";

    const res = await fetch("/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ username, password })
    });

    if (!res.ok) {
      error = "Invalid username or password";
      return;
    }

    location.reload();
  }
</script>

<div class="login-panel">
  <h3 class="login-title">Login</h3>

  <div class="login-body">
    <label>
      Username
      <input bind:value={username} autocomplete="username" />
    </label>

    <label>
      Password
      <input type="password" bind:value={password} autocomplete="current-password" />
    </label>

    <button class="login-btn" on:click={login}>Login</button>

    {#if error}
      <p class="login-error">{error}</p>
    {/if}
  </div>
</div>

<style>
  .login-panel {
    background: #111;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 1rem;
    width: 320px;
    margin: 4rem auto;
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace;
    color: #eee;
  }

  .login-title {
    margin: 0;
    padding-bottom: 0.25rem;
    font-size: 1rem;
    font-weight: bold;
    border-bottom: 1px solid #444;
  }

  .login-body {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 0.75rem;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  input {
    background: #222;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 0.4rem 0.5rem;
    color: #eee;
    font-family: inherit;
  }

  .login-btn {
    background: #333;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 0.5rem;
    color: #eee;
    cursor: pointer;
    font-family: inherit;
  }

  .login-btn:hover {
    background: #444;
  }

  .login-error {
    color: #f55;
    margin: 0;
    font-size: 0.85rem;
  }
</style>
