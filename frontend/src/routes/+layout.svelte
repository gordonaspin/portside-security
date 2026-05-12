<script lang="ts">
  import favicon from '$lib/assets/favicon.svg';
  export const ssr = false;
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
</svelte:head>

<slot />
