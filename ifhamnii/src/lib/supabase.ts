import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL  = "https://ndzuhaifxossbzenunzn.supabase.co"
const SUPABASE_KEY  = "sb_publishable_QtR-BIQZqfZydPBccLtrfA_0bzPBWZM"  // ← colle ta Publishable key ici

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY)